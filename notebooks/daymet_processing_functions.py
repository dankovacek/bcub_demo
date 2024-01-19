import os
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

#####
#
#  To extend the DAYMET dataset to add computed parameters, 
#  follow the examples below
#
#####
def compute_low_precip_frequency(ds, output_fpath, threshold=1.0):
    """
    Frequency of low precipitation days:
        -where precipitation < 1mm/day, or
    """
    #  write crs BEFORE AND AFTER resampling!
    non_nan_mask = ds.notnull()
    # count the number of dry days in each year
    # a dry day is one where precip = 0
    ds = (ds < threshold).where(non_nan_mask).resample(time='1Y', keep_attrs=True).sum('time', skipna=False) / 365.0
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    del non_nan_mask
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    mean.rio.write_crs(daymet_proj)
    mean.rio.write_nodata(np.nan, inplace=True)
    mean.rio.to_raster(output_fpath)
    return True


def consecutive_run_lengths(ds):
    # Find the change points
    # If any NaN values are present in the input array, return NaN
    if np.isnan(ds).all():
        return np.nan
    
    change = np.concatenate(([0], np.where(ds[:-1] != ds[1:])[0] + 1, [len(ds)]))
    # Calculate lengths and filter by True values
    lengths = np.diff(change)
    true_lengths = lengths[ds[change[:-1]]]
    # max_run = np.max(true_lengths) if true_lengths.size > 0 else 0
    mean_run = np.mean(true_lengths) if true_lengths.size > 0 else 0
    
    del ds
    del lengths

    return mean_run


def compute_low_prcp_duration(ds, output_fpath, threshold=1.0):
    # 1. Convert to a boolean array where True indicates values below the threshold
    nan_locations = ds.isnull().all(dim='time')
    below_threshold = (ds < threshold)
    below_threshold.rio.write_nodata(np.nan, inplace=True)
    below_threshold = below_threshold.rio.write_crs(daymet_proj)
    longest_runs = xr.apply_ufunc(consecutive_run_lengths, below_threshold.groupby('time.year'), 
                                  input_core_dims=[['time']], 
                                  vectorize=True, 
                                  dask='parallelized', 
                                  output_dtypes=[int])
    print('    finished computing longest runs')
    
    longest_runs = longest_runs.rio.write_crs(daymet_proj)
    # 3. Calculate the longest duration for each year
    
    # 4. Calculate the mean duration over all the years
    longest_runs = longest_runs.where(~nan_locations)
    
    mean_longest_run = longest_runs.mean('year', skipna=False, keep_attrs=True).round(1)
    
    mean_longest_run.rio.write_crs(daymet_proj)
    mean_longest_run.rio.write_nodata(np.nan, inplace=True)
    mean_longest_run.rio.to_raster(output_fpath)
    return True


def compute_high_precip_frequency(ds, output_fpath, threshold=5.0):
    # load the mean annual precip raster
    non_nan_mask = ds.notnull()
    # count the number of dry days in each year
    # a dry day is one where precip = 0
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    n_days = ds.time.size

    # find the frequency of days where the precip is greater than 5 x mean annual precip
    # by pixel-wise comparison, and divide by the length of the time dimension
    ds = (ds >= threshold * mean).where(non_nan_mask).sum('time', skipna=False) / n_days
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    ds.rio.to_raster(output_fpath)
    del non_nan_mask
    del mean
    del ds
    return True
    

def compute_high_precip_duration(ds, output_fpath, threshold=5.0):
    # 1. Convert to a boolean array where True indicates values below the threshold
    # non_nan_mask = ds.notnull()
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    
    above_threshold = (ds >= threshold * mean)
    nan_locations = ds.isnull().all(dim='time')
    del ds
    above_threshold.rio.write_nodata(np.nan, inplace=True)
    above_threshold = above_threshold.rio.write_crs(daymet_proj)
    print('    computing longest runs...')
    # 3. Calculate the duration of consecutive days >= 5x threshold
    longest_runs = xr.apply_ufunc(consecutive_run_lengths, above_threshold.groupby('time.year'), 
                                  input_core_dims=[['time']], 
                                  vectorize=True, 
                                  dask='parallelized', 
                                  output_dtypes=[int])
    print('    finished computing longest runs')
    del above_threshold

    longest_runs = longest_runs.rio.write_crs(daymet_proj)
    
    longest_runs = longest_runs.where(~nan_locations)    
    mean_longest_run = longest_runs.mean('year', skipna=False, keep_attrs=True).round(1)
    mean_longest_run.rio.write_crs(daymet_proj)
    mean_longest_run.rio.write_nodata(np.nan, inplace=True)
    mean_longest_run.rio.to_raster(output_fpath)
    return True


def set_computation_by_param(tid, param, output_fpath):
        # retrieve the precipitation data to compute statistics
        data = retrieve_tiles_by_id('prcp', tid)
        if param == 'low_prcp_freq':
            print(f'   Computing P(p<1mm) on {tid}')
            completed = compute_low_precip_frequency(data, output_fpath)
        elif param == 'low_prcp_duration':
            print(f'   Computing mean low precip duration')
            completed = compute_low_prcp_duration(data, output_fpath)
        elif param == 'high_prcp_freq':
            print(f'   Computing P(p >= 5 x mean annual)')
            completed = compute_high_precip_frequency(data, output_fpath)
        elif param == 'high_prcp_duration':
            print(f'   Computing mean high precip duration')
            completed = compute_high_precip_duration(data, output_fpath)
        else:
            print(f'No function set for processing parameter {param}')
            pass
        del data
        return completed


