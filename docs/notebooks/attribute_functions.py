import os
import time


# import pandas as pd
# import numpy as np
# import geopandas as gpd
# import rioxarray as rxr

# from scipy.stats.mstats import gmean

# import warnings
# warnings.simplefilter("ignore") 

# from utilities import retrieve_raster

# # this version date should reflect the number
# # associated with the derived basin set


# # porosity and permeability sources
# # # glhymps_fpath = os.path.join(ext_media_path, 'GLHYMPS/GLHYMPS.gdb')
# # glhymps_fpath = os.path.join(ext_media_path, 'GLHYMPS/GLHYMPS_clipped_4326.gpkg')


# # # import NALCMS raster
# # # land use / land cover
# # if not os.path.exists(reproj_nalcms_path):
# #     nalcms, nalcms_crs, nalcms_affine = retrieve_raster(nalcms_fpath)
# #     print(f'   ...NALCMS imported, crs = {nalcms.rio.crs.to_epsg()}')
# #     print('Reproject NALCMS raster')

# #     warp_command = f'gdalwarp -q -s_srs "{nalcms.rio.crs.to_proj4()}" -t_srs EPSG:4326 -of gtiff {nalcms_fpath} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS'    
# #     os.system(warp_command)

# # # 
# # print(f'Opening NALCMS raster:')
# # nalcms, nalcms_crs, nalcms_affine = retrieve_raster(reproj_nalcms_path)
# # print(f'   ...NALCMS imported, crs = {nalcms.rio.crs.to_epsg()}')

# # derived_basin_path = os.path.join(BASE_DIR, f'processed_data/processed_basin_polygons_{version}')

# # # dem_mosaic_path = os.path.join(ext_media_path, 'DEM_data/')
# # dem_mosaic_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
# # dem_mosaic_file = 'EENV_DEM_mosaic.vrt'
# # dem_mosaic_path = os.path.join(dem_mosaic_folder, dem_mosaic_file)

# # print(f'Opening {dem_mosaic_file}:')
# # srdem_raster, srdem_crs, srdem_affine = retrieve_raster(dem_mosaic_path)
# # print(f'   ...DEM imported, crs = {srdem_raster.rio.crs.to_epsg()}')
# # print(f'   ... {srdem_raster.rio.crs.to_proj4()}')
# # print('')


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


# def check_and_repair_geometries(in_feature):

#     # avoid changing original geodf
#     in_feature = in_feature.copy(deep=True)    
        
#     # drop any missing geometries
#     in_feature = in_feature[~(in_feature.is_empty)]
    
#     # Repair broken geometries
#     for index, row in in_feature.iterrows(): # Looping over all polygons
#         if row['geometry'].is_valid:
#             next
#         else:
#             fix = make_valid(row['geometry'])

#             try:
#                 in_feature.loc[[index],'geometry'] =  fix # issue with Poly > Multipolygon
#             except ValueError:
#                 in_feature.loc[[index],'geometry'] =  in_feature.loc[[index], 'geometry'].buffer(0)
#     return in_feature

    


# def get_soil_properties(merged, col):
#     # dissolve polygons by unique parameter values

#     geometries = check_and_repair_geometries(merged)

#     df = geometries[[col, 'geometry']].copy().dissolve(by=col, aggfunc='first')
#     df[col] = df.index.values
#     # re-sum all shape areas
#     df['Shape_Area'] = df.geometry.area
#     # calculuate area fractions of each unique parameter value
#     df['area_frac'] = df['Shape_Area'] / df['Shape_Area'].sum()
#     # check that the total area fraction = 1
#     total = round(df['area_frac'].sum(), 1)
#     sum_check = total == 1.0
#     if not sum_check:
#         print(f'    Area proportions do not sum to 1: {total:.2f}')
#         if np.isnan(total):
#             return np.nan
#         elif total < 0.9:
#             return np.nan
    
#     # area_weighted_vals = df['area_frac'] * df[col]
#     if 'Permeability' in col:
#         # calculate geometric mean
#         # here we change the sign (all permeability values are negative)
#         # and add it back at the end by multiplying by -1 
#         # otherwise the function tries to take the log of negative values
#         return gmean(np.abs(df[col]), weights=df['area_frac']) * -1
#     else:
#         # calculate area-weighted arithmetic mean
#         return (df['area_frac'] * df[col]).sum()
    

# def process_glhymps(basin_geo, fpath, new_proj):
#     # import soil layer with polygon mask (both in 4326)
#     # returns INTERSECTION
#     gdf = gpd.read_file(fpath, mask=basin_geo)
#     # now clip precisely to the basin polygon bounds
#     merged = gpd.clip(gdf, mask=basin_geo)
#     # now reproject to minimize spatial distortion
#     merged = merged.to_crs(new_proj)
#     return merged


# def process_basin_elevation(clipped_raster):
#     # evaluate masked raster data
#     values = clipped_raster.data.flatten()
#     mean_val = np.nanmean(values)
#     median_val = np.nanmedian(values)
#     min_val = np.nanmin(values)
#     max_val = np.nanmax(values)
#     return mean_val, median_val, min_val, max_val


# def calculate_gravelius_and_perimeter(polygon):    
#     perimeter = polygon.geometry.length.values[0]
#     area = polygon.geometry.area.values[0] 
#     if area == 0:
#         return np.nan, perimeter
#     else:
#         perimeter_equivalent_circle = np.sqrt(4 * np.pi * area)
#         gravelius = perimeter / perimeter_equivalent_circle
    
#     return gravelius, perimeter / 1000


# # @jit(nopython=True)
# def process_slope_and_aspect(E, el_px, resolution, shape):
#     # resolution = E.rio.resolution()
#     # shape = E.rio.shape
#     # note, distances are not meaningful in EPSG 4326
#     # note, we can either do a costly reprojection of the dem
#     # or just use the approximate resolution of 90x90m
#     # dx, dy = 90, 90# resolution
#     dx, dy = resolution
#     # print(resolution)
#     # print(asdfd)
#     # dx, dy = 90, 90
#     S, A = np.empty_like(E), np.empty_like(E)
#     S[:] = np.nan # track slope (in degrees)
#     A[:] = np.nan # track aspect (in degrees)
#     # tot_p, tot_q = 0, 0
#     for i, j in el_px:
#         if (i == 0) | (j == 0) | (i == shape[0]) | (j == shape[1]):
#             continue
            
#         E_w = E[i-1:i+2, j-1:j+2]

#         if E_w.shape != (3,3):
#             continue

#         a = E_w[0,0]
#         b = E_w[1,0]
#         c = E_w[2,0]
#         d = E_w[0,1]
#         f = E_w[2,1]
#         g = E_w[0,2]
#         h = E_w[1,2]
#         # skip i and j because they're already used
#         k = E_w[2,2]  

#         all_vals = np.array([a, b, c, d, f, g, h, k])

#         val_check = np.isfinite(all_vals)

#         if np.all(val_check):
#             p = ((c + 2*f + k) - (a + 2*d + g)) / (8 * abs(dx))
#             q = ((c + 2*b + a) - (k + 2*h + g)) / (8 * abs(dy))
#             cell_slope = np.sqrt(p*p + q*q)
#             S[i, j] = (180 / np.pi) * np.arctan(cell_slope)
#             A[i, j] = (180.0 / np.pi) * np.arctan2(q, p)

#     return S, A


# def calculate_circular_mean_aspect(a):
#     """
#     From RavenPy:
#     https://github.com/CSHS-CWRA/RavenPy/blob/1b167749cdf5984545f8f79ef7d31246418a3b54/ravenpy/utilities/analysis.py#L118
#     """
#     angles = a[~np.isnan(a)]
#     n = len(angles)
#     sine_mean = np.divide(np.sum(np.sin(np.radians(angles))), n)
#     cosine_mean = np.divide(np.sum(np.cos(np.radians(angles))), n)
#     vector_mean = np.arctan2(sine_mean, cosine_mean)
#     degrees = np.degrees(vector_mean)
#     if degrees < 0:
#         return degrees + 360
#     else:
#         return degrees


# def calculate_slope_and_aspect(raster):  
#     """Calculate mean basin slope and aspect 
#     according to Hill (1981).

#     Args:
#         clipped_raster (array): dem raster

#     Returns:
#         slope, aspect: scalar mean values
#     """
#     # print(raster.data[0])
#     # print(raster.rio.crs)
#     # print(asfd)
#     # wkt = raster.rio.crs.to_wkt()
#     # affine = raster.rio.transform()

#     resolution = raster.rio.resolution()
#     raster_shape = raster[0].shape

# #     rdem_clipped = rd.rdarray(
# #         raster.data[0], 
# #         no_data=raster.rio.nodata, 
# #         projection=wkt, 
# #     )

# #     rdem_clipped.geotransform = affine.to_gdal()
# #     rdem_clipped.projection = wkt

#     # # ts0 = time.time()
#     # use to check slope -- works out to within 1 degree...
#     # slope = rd.TerrainAttribute(rdem_clipped, attrib='slope_degrees')
#     # aspect_deg = rd.TerrainAttribute(rdem_clipped, attrib='aspect')
#     # # ts2 = time.time()

#     el_px = np.argwhere(np.isfinite(raster.data[0]))

#     # print(el_px[:2])
#     # print(rdem_clipped)
#     # print(asdfsd)
#     S, A = process_slope_and_aspect(raster.data[0], el_px, resolution, raster_shape)

#     mean_slope_deg = np.nanmean(S)
#     # should be within a hundredth of a degree or so.
#     # print(f'my slope: {mean_slope_deg:.4f}, rdem: {np.nanmean(slope):.4f}')
#     mean_aspect_deg = calculate_circular_mean_aspect(A)

#     return mean_slope_deg, mean_aspect_deg


# def check_lulc_sum(stn, data):
#     checksum = sum(list(data.values())) 
#     lulc_check = 1-checksum
#     if abs(lulc_check) >= 0.05:
#         print(f'   ...{stn} failed checksum: {checksum:.3f}')   
#     return lulc_check


# def recategorize_lulc(data):    
#     forest = ('Land_Use_Forest_frac', [1, 2, 3, 4, 5, 6])
#     shrub = ('Land_Use_Shrubs_frac', [7, 8, 11])
#     grass = ('Land_Use_Grass_frac', [9, 10, 12, 13, 16])
#     wetland = ('Land_Use_Wetland_frac', [14])
#     crop = ('Land_Use_Crops_frac', [15])
#     urban = ('Land_Use_Urban_frac', [17])
#     water = ('Land_Use_Water_frac', [18])
#     snow_ice = ('Land_Use_Snow_Ice_frac', [19])
#     lulc_dict = {}
#     for label, p in [forest, shrub, grass, wetland, crop, urban, water, snow_ice]:
#         prop_vals = round(sum([data[e] if e in data.keys() else 0.0 for e in p]), 2)
#         lulc_dict[label] = prop_vals
#     return lulc_dict
    

# def get_value_proportions(data, new_proj):
#     # vals = data.data.flatten()
#     all_vals = data.data.flatten()
#     vals = all_vals[~np.isnan(all_vals)]
#     n_pts = len(vals)
#     unique, counts = np.unique(vals, return_counts=True)
#     # create a dictionary of land cover values by coverage proportion
#     # assuming raster pixels are equally sized, we can keep the
#     # raster in geographic coordinates and just count pixel ratios
#     prop_dict = {k: 1.0*v/n_pts for k, v in zip(unique, counts)}
#     if prop_dict[15] > 0.01:
#         print(prop_dict)
#         print(asdf)
#     prop_dict = recategorize_lulc(prop_dict)
#     return prop_dict    


# def process_lulc(stn, basin_geo, nalcms_crs):
#     # polygon = basin_polygon.to_crs(nalcms_crs)
#     # assert polygon.crs == nalcms.rio.crs
#     assert basin_geo.crs == nalcms.rio.crs
#     raster_loaded, lu_raster_clipped = clip_raster_to_basin(basin_geo, nalcms)
#     # checksum verifies proportions sum to 1
#     prop_dict = get_value_proportions(lu_raster_clipped, new_proj)
#     lulc_check = check_lulc_sum(stn, prop_dict)
#     prop_dict['lulc_check'] = lulc_check
#     return pd.DataFrame(prop_dict, index=[stn])


# def compare_areas(stn, areas, names):

#     print(areas, names)

#     hysets_area = hysets_df[hysets_df['Official_ID'] == stn]['Drainage_Area_km2'].values[0]
#     # print('hysets_area')
#     # print(hysets_area)
#     diffs = []
#     for i in range(len(names)):
#         diff = 1 - areas[i] / hysets_area
#         if (abs(diff) > 0.05) & (names[i] != '4326'):
#             print(f'BIG AREA DIFF!!')
#             print(f'{stn} diff {diff}  ({names[i]})')
#             # raise Exception('Big area diff!')



# def process_basin_characteristics(stn, reproj_raster, c, basin_geo, temp_raster_path, new_proj):
#     print(f'    ...processing {stn} basin characteristics in {c}')
#     # print(temp_raster_path)
#     basin_data = {}
#     # print(f'   Processing {stn}')
#     basin_data['Official_ID'] = str(stn)

#     base_path = '/'.join(temp_raster_path.split('/')[:-1])

#     if c == 'LAEA':
#         # eq_epsg = pyproj.CRS(new_proj).to_epsg(min_confidence=0)
#         # print(eq_epsg)
#         basin_proj = basin_geo.to_crs(new_proj)
#         # basin_proj.to_file(os.path.join(base_path, f'basin_LAEA.geojson'))
#     else:
#         basin_proj = basin_geo.to_crs(c)
#         # basin_proj.to_file(os.path.join(base_path, f'basin_{c}.geojson'))
    
#     print('    Reprojected polygon.')

#     hysets_data = hysets_stn_df[hysets_stn_df['Official_ID'] == stn]
#     hs_slope = hysets_data['Slope_deg'].values[0]
#     hs_asp = hysets_data['Aspect_deg'].values[0]
#     hs_el = hysets_data['Elevation_m'].values[0]
#     hs_area = hysets_data['Drainage_Area_km2'].values[0]
#     hs_grav = hysets_data['Gravelius'].values[0]

#     basin_data['Val_Drainage_Area_km2'] = basin_proj.geometry.area[0] / 1E6
#     # Drainage area, perimeter, and gravelius tested against independent functions
    
#     geom_props = raven_analysis.geom_prop(basin_proj.geometry.values[0])
    
#     basin_data['Raven_Drainage_Area_km2'] = geom_props['area'] / 1E6
    
#     basin_data['Raven_Perimeter'] = geom_props['perimeter'] / 1E3
#     basin_data['Raven_Gravelius'] = geom_props['gravelius']

#     # projected_polygon = basin_polygon.to_crs(proj_crs)
#     gravelius, perimeter = calculate_gravelius_and_perimeter(basin_proj)
#     # print(f'gravelius, perimeter: {gravelius:.2f} {perimeter:.0f} ')
#     basin_data['Val_Perimeter'] = perimeter
#     basin_data['Val_Gravelius'] = gravelius

#     # ta = time.time()
    
#     # mean_el, median_el, _, _ = process_basin_elevation(reproj_raster)
#     # basin_data['median_el'] = median_el
#     # basin_data['mean_el'] = mean_el
    
#     slope, aspect = calculate_slope_and_aspect(reproj_raster)
#     # print(f'aspect, slope: {aspect:.1f} {slope:.2f} ')
#     basin_data['Val_Slope_deg'] = slope
#     basin_data['Val_Aspect_deg'] = aspect
#     # tb = time.time()
#     # t_mine = tb-ta
#     # my_area = basin_proj.geometry.area[0] / 1E6
#     # rv_area = basin_data['Drainage_Area_km2']
#     # print(f'    MY     el: {mean_el:.2f}, slp: {slope:.2f}, aspect: {aspect:.2f}, area: {my_area:.2f} km2')
#     # print(f'    My el/sope/aspect time: {t_mine:.3f}')

#     # ta = time.time()
#     # print(temp_raster_path)
#     # print('starting raven dem prop analysis')
#     dem_props = raven_analysis.dem_prop(temp_raster_path, basin_proj.geometry[0])
#     # print(dem_props)
#     basin_data['Elevation_m'] = dem_props['elevation']
#     basin_data['Raven_Slope_deg'] = dem_props['slope']
#     basin_data['Raven_Aspect_deg'] = dem_props['aspect']
#     # el, slp, asp = dem_props['elevation'], dem_props['slope'], dem_props['aspect']
#     # tb = time.time()
#     # t_mine = tb-ta
#     # print(f'    RAVEN  el: {el:.2f}, slp: {slp:.2f}, aspect: {asp:.2f},  area: {rv_area:.2f} km2 ')
#     # print(f'    Raven el/sope/aspect time: {t_mine:.3f}s')    
#     # print(f'    HYSETS el: {hs_el:.2f}, slp: {hs_slope:.2f}, aspect: {hs_asp:.2f}, {hs_area:.2f} km2 ')
#     # print('')

#     # process lulc
#     lulc_df = process_lulc(stn, basin_geo, nalcms_crs)
#     lulc_data = lulc_df.to_dict('records')[0]
#     # print('     lulc complate')
#     basin_data.update(lulc_data)
#     # lu_cols = [e for e in hysets_df.columns if 'Land_Use' in e]
#     # lu_cols = [e for e in lu_cols if 'Flag' not in e]
#     # hys_lu_vals = hysets_df[hysets_df['Official_ID'] == stn_id][lu_cols]

    
#     # area = projected_polygon.area.values[0] / 1E6
#     # print('mine:')
#     # print(f'area: {area}, perim: {perimeter}, grav: {gravelius}')
#     # print(perimeter, gravelius, area)
#     # print(asfsd)
#     soil_cols = ['Permeability_logk_m2', 'Porosity_frac']
#     # hys_gl_vals = hysets_df[hysets_df['Official_ID'] == stn_id][soil_cols]
#     # print(hys_gl_vals)
#     # print('starting glhymps')
#     # glhymps_df = process_glhymps(basin_geo)
#     glhymps_projected  = process_glhymps(basin_geo, glhymps_fpath, new_proj)
#     # weighted_permeability, weighted_porosity = get_perm_and_porosity(glhymps_projected)
#     porosity = get_soil_properties(glhymps_projected, 'Porosity')
#     permeability = get_soil_properties(glhymps_projected, 'Permeability_no_permafrost')
#     basin_data['Permeability_logk_m2'] = permeability
#     basin_data['Porosity_frac'] = porosity
#     # hys_lu_vals = hysets_df[hysets_df['Official_ID'] == stn_id][lu_cols]
#     # print(f' perm: {permeability:.2f} por: {porosity:.2f}')

#     # print(f'DA: {area:.1f} ({hysets_area:.1f}) el {mean_el:.0f} ({hysets_el:.0f}) slope: {slope:.1f} ({hysets_slope:.1f}) aspect: {aspect:.1f} ({hysets_aspect:.1f}) grav: {gravelius:.1f} ({hysets_grav:.1f}) perim. {perimeter:.0f} ({hysets_perim:.0f}) ')

#     return basin_data


# def warp_raster(stn, new_proj, c, temp_dem_folder, temp_raster_path_in):
    
#     if c != 'LAEA':
#         t_srs = f'EPSG:{c}'
#         proj_code = c
#     else:
#         t_srs = f"'{new_proj}'" 
#         proj_code = 'LAEA'
    
#     temp_raster_path_out = os.path.join(temp_dem_folder, f'{stn}_temp_{proj_code}.tif')

#     warp_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs {t_srs} -of gtiff {temp_raster_path_in} {temp_raster_path_out} -wo NUM_THREADS=ALL_CPUS'    

#     try:
#         # print(warp_command)
#         os.system(warp_command)            
#         return True, temp_raster_path_out
#     except Exception as ex:
#         print('')
#         print(f'Raster reprojection failed for {stn}.')
#         print('')
#         print(ex)
#         print('')
#         return False, None


# def update_results_files(all_data, update):
#     basin_characteristics = pd.DataFrame(all_data)

#     if update:
#         updated_results = pd.concat([results_df, basin_characteristics])
#         n_results = len(updated_results)
#         print(f'Results updated.  Completed {n_results}/14425.',)

#     else:
#         print('new results!!')
#         updated_results = basin_characteristics

#     updated_results.to_csv(results_path, index=False)
#     return updated_results


# # 'derived' (validation) or 'baseline' (HYSETS)
# which_set = 'derived'
# # which_set = 'hysets'

# derived_basin_polygons = list(set([e for e in os.listdir(derived_basin_path) if e.endswith(f'_{which_set}.geojson')]))

# # basins_to_process = [e.split('_')[0] for e in derived_basin_polygons]

# t0 = time.time()

# results_path = os.path.join(BASE_DIR, f'results/{which_set}_characteristics_{version}_RE.csv')

# update = False
# if os.path.exists(results_path):
#     results_df = pd.read_csv(results_path)
#     processed_stns = results_df['Official_ID'].astype(str).values
#     update = True
#     basins_to_process = [e for e in derived_basin_polygons if e.split('_')[0] not in processed_stns]
# else:
#     basins_to_process = derived_basin_polygons


# all_data = []
# n = 0
# t0 = time.time()

# print(f'There are {len(basins_to_process)} remaining basins to be processed.')
# print('')
# print(basins_to_process)
# print(sdfasdf)
# # hysets_ids = hysets_stn_df['OfficialID'].values
# n = 0
# for polygon_file in basins_to_process: 
#     stn_id = polygon_file.split('_')[0]
               
#     print(f'Starting basin processing on {stn_id}.')

#     polygon_path = os.path.join(derived_basin_path, polygon_file)
#     basin_polygon = gpd.read_file(polygon_path)
#     basin_geo = basin_polygon.to_crs(4326)
#     # basin_geo = gpd.GeoDataFrame(geometry=[row['geometry']], crs='EPSG:4326')

#     centroid = basin_geo.geometry.centroid
#     lat, lon = centroid.y.values[0], centroid.x.values[0]
#     # new_proj = f'+proj=laea +lat_0={lat:.5f} +lon_0={lon:.5f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True'
#     new_proj = f'+proj=laea +lat_0={lat:.5f} +lon_0={lon:.5f} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

#     # print(new_proj)

#     temp_dem_folder = os.path.join(BASE_DIR, 'temp/')
#     if not os.path.exists(temp_dem_folder):
#         os.mkdir(temp_dem_folder)

#     # temp_raster_path_in = os.path.join(temp_dem_folder, f'{stn_id}_temp_raster.tif')
#     # temp_geom_path = basin_geo.to_file(os.path.join(temp_dem_folder, f'{stn_id}_basin.geojson'))

#     tx = time.time()
#     # clip raster to basin polygon
#     # assert basin_polygon.crs == srdem_raster.rio.crs
    
#     # clipped_raster = raven_geo.generic_raster_clip(
#     #     # raster=srdem_raster, 
#     #     raster=dem_mosaic_path,
#     #     output=temp_raster_path_in,
#     #     geometry=temp_geom_path,
#     #     )
#     # this produces exact match of output from raven_geo.generic_raster_clip()

#     # temp_raster_path_in2 = os.path.join(temp_dem_folder, f'{stn_id}_temp_raster2.tif')
#     # clip the raster using the basin where both are in a geographic coordinate reference system (EPSG:4326)
#     raster_loaded, clipped_raster = clip_raster_to_basin(basin_geo, srdem_raster)
#     temp_raster_path_in = os.path.join(temp_dem_folder, f'{stn_id}_temp_4326.tif')
#     clipped_raster.rio.to_raster(temp_raster_path_in)
    
#     for c in ['LAEA']:#, 3978, 3005]:
        
#         ty = time.time()
#         # print(f'    ...raster clip time: {ty-tx:.2f}')

#         # print(f'    ...raster loaded and clipped: {raster_loaded}')

#         # to minimize distortion for the area calculation,
#         # warp the polygon to lambert azimuthal equal area 
#         # oriented about the basin centroid
#         raster_warped, temp_raster_path = warp_raster(stn_id, new_proj, c, temp_dem_folder, temp_raster_path_in)
    
#         # if not raster_warped:
#         #     print(f'Raster warp failed for {stn_id}')
#         #     continue
#         # else:
#         reproj_raster, warped_crs, warped_affine = retrieve_raster(temp_raster_path)
        
#         basin_attributes = process_basin_characteristics(stn_id, reproj_raster, c, basin_geo, temp_raster_path, new_proj)

#         all_data.append(basin_attributes)
    
    
#     n += 1
#     if (n % 100 == 0) | (n > 14300):
#         t_end = time.time()
#         unit_time = (t_end-t0) / n
#         updated_df = update_results_files(all_data, update)
#         print(f'Processed basin {n}/{len(updated_df)} in {t_end-t0:.0f}s ({unit_time:.2f}s/basin)')

#         temp_files = os.listdir(temp_dem_folder)
#         for f in temp_files:
#             os.remove(os.path.join(temp_dem_folder, f))
#         print('    ...temporary files removed.')
#         print('')



