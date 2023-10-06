import os
import time
import random
import numpy as np
import pandas as pd
import rioxarray as rxr
from shapely.geometry import Point, LineString
import geopandas as gpd
import multiprocessing as mp


def clip_raster_to_basin(clipping_polygon, raster):
    bounds = tuple(clipping_polygon.bounds.values[0])
    try:
        # clip first to bounding box, then to polygon for better performance
        subset_raster = raster.rio.clip_box(*bounds).copy()
        clipped_raster = subset_raster.rio.clip(
        # clipped_raster = raster.rio.clip(
            clipping_polygon.geometry, 
            clipping_polygon.crs.to_epsg(),
            all_touched=True,
            )
        return True, clipped_raster
    except Exception as e:
        print(e)
        return False, None

def retrieve_raster(filename):
    """
    Take in a file name and return the raster data, 
    the coordinate reference system, and the affine transform.
    """
    raster = rxr.open_rasterio(filename, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    return raster, crs.to_epsg(), affine


def mask_flow_direction(S_w, F_w):  
    """
    Get all the neighbouring stream cells 
    pointing towards the target (centre) cell.
    """

    inflow_dirs = np.array([
        [4, 8, 16],
        [2, 0, 32],
        [1, 128, 64]
    ])
    # mask the flow direction by element-wise multiplication
    # using the boolean stream matrix
    F_sm = np.multiply(F_w, S_w)

    F_m = np.where(F_sm == inflow_dirs, True, False)

    return F_m


def check_for_confluence(i, j, ppts, S_w, F_m):
    """
    Check for confluences:
    --> If the flow direction of > 1 outer cell
    points towards the centre cell.
    """
    F_mc = F_m.copy()
    F_mc[1, 1] = 1

    # get inflow cell indices
    inflow_cells = np.argwhere(F_mc)

    fp_idx = f'{i},{j}'

    if len(inflow_cells) <= 2:
        # don't overwrite an already checked cell
        if 'CONF' not in ppts[fp_idx]:
            ppts[fp_idx]['CONF'] = False        
    else:
        ppts[fp_idx]['CONF'] = True        

        for (ci, cj) in inflow_cells:
            ix = ci + i - 1
            jx = cj + j - 1

            pt_idx = f'{ix},{jx}'

            # cells flowing into the focus 
            # cell are confluence cells
            if pt_idx not in ppts:
                ppts[pt_idx] = {'CONF': True}
            else:
                ppts[pt_idx]['CONF'] = True

    return ppts


def create_pour_point_gdf(region, stream, ppt_df, crs, output_path):
    """Convert the dictionary of confluences (raster indices) into a geodataframe
    of geographic points.  Break apart the list of stream pixels to avoid memory 
    allocation issue when indexing large rasters.

    Args:
        stream (_type_): _description_
        confluences (_type_): _description_
        n_chunks (int, optional): _description_. Defaults to 2.

    Returns:
        geodataframe: Dataframe of confluence points.
    """    
    n_chunks = int(10 * np.log(len(ppt_df)))
    print(f'creating {n_chunks} chunks for processing')

    # split the dataframe into chunks 
    # because calling coordinates of the raster
    # from a large array seems to be memory intensive.
    conf_chunks = np.array_split(ppt_df, n_chunks)
    processed_chunks = []
    ta = time.time()
    n = 0
    for chunk in conf_chunks:
        n += 1
        ppts = stream[0, chunk['ix'].values, chunk['jx'].values]
        coords = tuple(map(tuple, zip(ppts.coords['x'].values, ppts.coords['y'].values)))
        chunk['geometry'] = [Point(p) for p in coords]
        processed_chunks.append(chunk)
        tb = time.time()
        if n % 10 == 0:
            print(f'    ...{n}/{n_chunks} chunks processed in {tb-ta:.1f}s')

    gdf = gpd.GeoDataFrame(pd.concat(processed_chunks), crs=crs)
    print(f'    {len(gdf)} pour points created.') 
    t1 = time.time()
    print(f'   ...ppts geodataframe processed in{t1-ta:.1f}s\n')
    return gdf


def redistribute_vertices(geom, distance):
    """
    Here we resample the vector polygon geometry in order to
    calculate distances to points in the stream network.
    """
    if geom.geom_type in ['LineString', 'LinearRing']:
        # resample the polygon to have a number of vertices
        # to match the DEM resolution
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode(index_parts=False).reset_index(drop=True).geometry.values
        parts = [redistribute_vertices(part, distance)
                 for part in geoms]
        print(parts)
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def format_batch_geometries(all_polygons, ppt_batch, region_raster_crs):
    ppt_batch['ppt_x'] = ppt_batch.geometry.x.astype(int)
    ppt_batch['ppt_y'] = ppt_batch.geometry.y.astype(int)
    ppt_batch.drop(inplace=True, labels=['geometry'], axis=1)
    batch_polygons = gpd.GeoDataFrame(pd.concat(all_polygons), crs=region_raster_crs)
    batch_polygons.sort_values(by='VALUE', inplace=True)
    batch_polygons.reset_index(inplace=True, drop=True)
    batch = gpd.GeoDataFrame(pd.concat([batch_polygons, ppt_batch], axis=1), crs=region_raster_crs)
    batch['centroid_x'] = batch.geometry.centroid.x.astype(int)
    batch['centroid_y'] = batch.geometry.centroid.y.astype(int)
    return batch    


def create_ppt_file_batches(df, filesize, temp_ppt_filepath, batch_limit=1E6):    
    """
    divide the dataframe into chunks for batch processing
    save the pour point dataframe to temporary filRes
    and limit temporary raster files to ?GB / batch
    """
    n_batches = int(filesize * len(df) / batch_limit) + 1
    print(f'        ...running {n_batches} batch(es) on {filesize:.1f}MB raster.')

    include_cols = ['acc', 'ix', 'jx', 'geometry']
    
    df = df[include_cols]
    
    # batch_paths, idx_batches = [], []
    batch_paths = []
    n = 0
    if len(df) * filesize < batch_limit:
        temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
        df.to_file(temp_fpath)
        batch_paths.append(temp_fpath)
        # idx_batches.append(df.index.values)
    else:
        # randomly shuffle the indices 
        # to split into batches
        indices = df.index.values
        random.shuffle(indices)
        batches = np.array_split(np.array(indices), indices_or_sections=n_batches)
        for batch in batches:
            n += 1
            batch_gdf = df.iloc[batch].copy()
            # idx_batches.append(batch)
            temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
            batch_gdf.to_file(temp_fpath)
            # keep just the filename
            batch_paths.append(temp_fpath)

    # return list(zip(batch_paths, idx_batches))
    return batch_paths


def clean_up_temp_files(temp_folder, batch_rasters):    
    """
    Between batches we need to clean up temporary files.
    """
    temp_files = [f for f in os.listdir(temp_folder) if 'temp_' in f]
    for f in list(set(temp_files)):
        os.remove(os.path.join(temp_folder, f))
        

def check_for_ppt_batches(batch_folder):    
    """
    If we encounter issues in the basin delineation process, we 
    can pick up where we left off by checking for existing batch files.
    """
    if not os.path.exists(batch_folder):
        os.mkdir(batch_folder)
        return False
    existing_batches = os.listdir(batch_folder)
    return len(existing_batches) > 0
    

def filter_and_explode_geoms(gdf, min_area):
    gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
    gdf = gdf.explode(index_parts=True)
    gdf['area'] = gdf.geometry.area / 1E6
    gdf = gdf[gdf['area'] >= min_area * 0.95]    
    return gdf


def match_ppt_to_polygons_by_order(ppt_batch, polygon_df, resolution):
    """
    We need to match the input pour points to the basin polygons.
    Here we take the input pour point (batch) shapefile, the basin polygon 
    geodataframe resulting from converting raster to vector polygons.
    We also take theraster resolution such that we can derive the upstream
    accumulation and validate against the polygon geometry.
    """

    try:
        assert len(ppt_batch) == len(polygon_df)
    except Exception as e:
        print(f' mismatched df lengths: ppt vs. polygon_df')
        print(len(ppt_batch), len(polygon_df))
        print('')

    polygon_df['acc_polygon'] = (polygon_df.geometry.area / (resolution[0] * resolution[1])).astype(int)
    polygon_df['ppt_acc'] = ppt_batch['acc'].values
    polygon_df['acc_diff'] = polygon_df['ppt_acc'] - polygon_df['acc_polygon']
    polygon_df['FLAG_acc_match'] = polygon_df['acc_diff'].abs() > 2

    polygon_df['ppt_x'] = ppt_batch.geometry.x.values
    polygon_df['ppt_y'] = ppt_batch.geometry.y.values
    polygon_df['cell_idx'] = ppt_batch['cell_idx'].values

    polygon_df.reset_index(inplace=True, drop=True)

    return polygon_df


def interpolate_line(inputs):
    geom, n, num_vertices = inputs
    d = n / num_vertices
    return (n, geom.interpolate(d, normalized=True))
    

def redistribute_vertices(geom, distance):
    """Evenly resample along a linestring
    See this SO post:
    https://gis.stackexchange.com/a/367965/199640
    
    Args:
        geom (polygon): lake boundary geometry
        distance (numeric): distance between points in the modified linestring

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if geom.geom_type in ['LineString', 'LinearRing']:
        num_vertices = int(round(geom.length / distance))
        
        if num_vertices == 0:
            num_vertices = 1
        # print(f'total distance = {geom.length:.0f} m, n_vertices = {num_vertices}')
        inputs = [(geom, float(n), num_vertices) for n in range(num_vertices + 1)]
        with mp.Pool() as pool:
            results = pool.map(interpolate_line, inputs)
        
        df = pd.DataFrame(results, columns=['n', 'geometry'])
        df = df.sort_values(by='n').reset_index(drop=True)
        return LineString(df['geometry'].values)
    
    elif geom.geom_type == 'MultiLineString':
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode().reset_index(drop=True).geometry.values
        parts = [redistribute_vertices(part, distance)
                 for part in geoms]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))
