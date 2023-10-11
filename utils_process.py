import os
import gc
import glob
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import rasterio
import rasterstats
import geopandas as gpd
from osgeo import gdal
from tqdm.notebook import tqdm
import cftime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


def calc_zonal(root:str, gdf: gpd.GeoDataFrame, variable: str, mode: str) -> pd.DataFrame:
    """
    This function calculates the chosen zonal statistics of the chosen variable based on the inputted shapefile.
    The output is a concatenated dataframe of the zonal statistics timeseries

    Keyword arguments:
    root -- string of the parent directory path
    gdf -- the shapefile as a geodataframe
    variable -- string of the chosen variable
    mode -- string of the chosen mode ['mean', 'max', ...]

    """

    list_paths = glob.glob(os.path.join(root, f'raw_data/{variable.lower()}/*.tif'))
    list_paths.sort()

    poly_lst = []

    for path in tqdm(list_paths):

        ind = path.rfind('_')

        if variable == 'luminosity':
            ind = path.rfind('bm_')+2

        try:
            ds = rasterio.open(path)
            arr = ds.read(1)
            affine = ds.transform

            df_poly = gdf.copy(deep=True)
            zonal = rasterstats.zonal_stats(df_poly.geometry, arr, affine=affine, stats=mode)
            # zonal = rasterstats.zonal_stats(df_poly.geometry, path, stats=mode)
            df_poly[variable] = pd.DataFrame(zonal)
            df_poly['date'] = path[ind+1:ind+11]
            df_poly['year'] = path[ind+1:ind+5]

            if variable in ['co', 'no2', 'o3', 'pm25', 'so2']:
                df_poly = df_poly.rename({variable: mode}, axis=1)
                df_poly['type'] = variable.upper()
                df_poly = df_poly.drop(['year'], axis=1)


            poly_lst.append(df_poly)
        except Exception as e:
            continue

        del ds, arr, df_poly

    gc.collect()
    
    return pd.concat(poly_lst).reset_index(drop=True).drop(['geometry'], axis=1)


def calc_zonal_land_cover(root: str, gdf: gpd.GeoDataFrame, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the categorical zonal statistics of the inputted geometry's land cover. 
    The output is a concatenated dataframe of land cover timeseries

    Keyword arguments:
    root -- string of the parent directory path
    gdf -- the chosen shapefile as a geodataframe
    admin_level -- integer of the chosen administrative level 
    """

    cmap = {0: 'water', 1: 'trees', 2: 'grass', 3: 'flooded_vegetation', 4: 'crops', 5: 'shrub_and_scrub', 
        6: 'built', 7: 'bare', 9: 'snow_and_ice'}
    crs = 'EPSG:4326'

    list_paths = glob.glob(os.path.join(root, 'raw_data/land_cover/*.tif'))
    list_paths.sort()

    poly_lst = list()
    
    for path in tqdm(list_paths):

        ind = path.rfind('_')

        df_poly = gdf.copy(deep=True)
        df_poly = df_poly.to_crs(crs)
        mean_tree = rasterstats.zonal_stats(df_poly.geometry, 
                                            path,
                                            categorical=True, category_map=cmap)
        
        df_poly['year'] = path[ind+1:ind+5]

        df = pd.DataFrame(mean_tree).fillna(0)
        df['green_pct'] = 100*(df['trees']+df['grass']+df['shrub_and_scrub']) / df.sum(axis=1)
        df['built_pct'] = 100*df['built']/df.sum(axis=1)
        df_poly = pd.concat([df_poly, df[['green_pct', 'built_pct']]], axis=1)

        poly_lst.append(df_poly)
    
    df_land_cover = pd.concat(poly_lst, axis=0).drop(['geometry'], axis=1)
    df_land_cover = df_land_cover.sort_values(by=[f'GID_{admin_level}', 'year']).reset_index(drop=True)
    df_land_cover['green_cover_growth'] = df_land_cover.groupby([f'GID_{admin_level}'])['green_pct'].pct_change() * 100
    df_land_cover['built_cover_growth'] = df_land_cover.groupby([f'GID_{admin_level}'])['built_pct'].pct_change() * 100

    return df_land_cover

def calc_zonal_pop(root: str, gdf: gpd.GeoDataFrame, growth_rates: list, country: str) -> pd.DataFrame: 
    """
    This function calculates the Worldpop population zonal statistics. Since Worldpop's data after 2020 is missing, the annual growth rates 
    are estimated and inputted manually. This function is subject to change if a more complete alternative dataset is used instead.
    It outputs a concatenated pandas dataframe of population data timeseries

    Keyword Arguments:
    root -- string of the parent directory path
    gdf -- the input shapefile as a geodataframe
    growth_rates -- list of 2021 and 2022's annual population growth rates
    country -- string of the chosen country
    """
    crs = 'EPSG:4326'
    poly_lst = list()

    for i in range(2018, 2021):
        print(i)
        df_poly = gdf.copy(deep=True)

        df_poly = df_poly.to_crs(crs)
        pop = rasterstats.zonal_stats(df_poly.geometry, 
                                    os.path.join(root, f'raw_data/population/{country.lower()}_ppp_{i}_1km_Aggregated.tif'),
                                    stats='sum')
    
        df_poly['population'] = pd.DataFrame(pop)
        df_poly['year'] = i
        
        poly_lst.append(df_poly)

    df_pop = pd.concat(poly_lst, axis=0).drop(['geometry'], axis=1).reset_index(drop=True)

    df_pop_2021 = df_pop[df_pop['year'] == 2020]
    df_pop_2021['year'] = 2021
    df_pop_2021['population'] = df_pop_2021['population'] * growth_rates[0]

    df_pop_2022 = df_pop_2021.copy(deep=True)
    df_pop_2022['year'] = 2022
    df_pop_2022['population'] = df_pop_2022['population'] * growth_rates[1]

    df_pop = pd.concat([df_pop, df_pop_2021, df_pop_2022], axis=0).reset_index(drop=True)

    return df_pop


def get_PM25_subindex(x: float) -> float:
    """
    This function calculates the PM25 subindex using the EPA's formula

    Keyword Arguments:
    x -- float of the PM25's value
    """
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0
    
def get_NO2_subindex(x: float) -> float:
    """
    This function calculates the NO2 subindex using the EPA's formula

    Keyword Arguments:
    x -- float of the NO2's value
    """
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

def get_CO_subindex(x):
    """
    This function calculates the CO subindex using the EPA's formula

    Keyword Arguments:
    x -- float of the CO's value
    """
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0
    
def get_SO2_subindex(x):
    """
    This function calculates the SO2 subindex using the EPA's formula

    Keyword Arguments:
    x -- float of the SO2's value
    """ 
    if   x < 0:
        return 0
    elif x<= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800

    else:
        return 0

def get_O3_subindex(x):
    """
    This function calculates the O3 subindex using the EPA's formula

    Keyword Arguments:
    x -- float of the O3's value
    """ 
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0
    
#define qualitative air quality index generator funtion
def get_AQI_bucket(x):
    """
    This function calculates the AQI bucket using the EPA's guidelines

    Keyword Arguments:
    x -- float of the AQI's value
    """
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN
    
def calc_aqi(aqi: pd.DataFrame, admin_level: int) -> pd.DataFrame:
    """
    This function calculates the AQI scores and buckets for all air pollutants based on the EPA's guidelines.

    Keyword Argument:
    aqi -- AQI pandas dataframe
    admin_level -- string of the chosen administrative level
    """

    aqi = aqi[aqi['date'] >= '2018-01-01']
    df_AQI = aqi.pivot_table(index=['date', f'GID_{admin_level}'], columns='type', values='mean').reset_index()
    df_AQI = df_AQI.fillna(0)

    df_AQI['PM25'] = df_AQI['PM25'] * (10 ** 9)
    df_AQI['PM2.5_SubIndex'] = df_AQI['PM25'].apply(lambda x: get_PM25_subindex(x))

    df_AQI['NO2'] = df_AQI['NO2'] * (10 ** 6)
    df_AQI['NO2_SubIndex'] = df_AQI['NO2'].apply(lambda x: get_NO2_subindex(x))

    df_AQI['CO_SubIndex'] = df_AQI['CO'].apply(lambda x: get_CO_subindex(x))

    df_AQI['SO2'] = df_AQI['SO2'] * (10 ** 3)
    df_AQI['SO2_SubIndex'] = df_AQI['SO2'].apply(lambda x: get_SO2_subindex(x))

    df_AQI['O3'] = df_AQI['O3'] * (10 ** 3)
    df_AQI['O3_SubIndex'] = df_AQI['O3'].apply(lambda x: get_O3_subindex(x))

    df_AQI["Checks"] = (df_AQI["PM2.5_SubIndex"] > 0).astype(int) + \
                (df_AQI["SO2_SubIndex"] > 0).astype(int) + \
                (df_AQI["NO2_SubIndex"] > 0).astype(int) + \
                (df_AQI["CO_SubIndex"] > 0).astype(int) + \
                (df_AQI["O3_SubIndex"] > 0).astype(int)

    df_AQI["AQI_calculated"] = round(df_AQI[["PM2.5_SubIndex", "SO2_SubIndex", "NO2_SubIndex",
                                    "CO_SubIndex", "O3_SubIndex"]].max(axis = 1, skipna=True))

    df_AQI.loc[df_AQI["PM2.5_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
    df_AQI.loc[df_AQI.Checks < 3, "AQI_calculated"] = np.NaN
    df_AQI = df_AQI[df_AQI['AQI_calculated'].notnull()]

    df_AQI['AQI_bucket'] = df_AQI['AQI_calculated'].apply(lambda x: get_AQI_bucket(x))

    # Convert the 'Date' column to datetime format
    df_AQI['date'] = pd.to_datetime(df_AQI['date'])
    # Create a new column for the year
    df_AQI['year'] = df_AQI['date'].dt.year

    return df_AQI

def calc_aqi_agg(df_AQI: pd.DataFrame, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the annual percentage change of the air pollutant indices.
    
    Keyword Arguments:
    df_AQI -- AQI pandas dataframe
    admin_level -- integer of chosen administrative level
    """

    df_AQI_gb = df_AQI.groupby([f'GID_{admin_level}', 'year']).agg(PM25_SUBINDEX=('PM2.5_SubIndex', 'mean'),
                                                    NO2_SUBINDEX=('NO2_SubIndex', 'mean'),
                                                    CO_SUBINDEX=('CO_SubIndex', 'mean'),
                                                    SO2_SUBINDEX=('SO2_SubIndex', 'mean'),
                                                    O3_SUBINDEX=('O3_SubIndex', 'mean')).reset_index()
    df_AQI_gb = df_AQI_gb.sort_values(by=[f'GID_{admin_level}', 'year']).reset_index(drop=True)
    df_AQI_gb['PM25_PCT_CHANGE'] = df_AQI_gb.groupby([f'GID_{admin_level}'])['PM25_SUBINDEX'].pct_change()*100
    df_AQI_gb['NO2_PCT_CHANGE'] = df_AQI_gb.groupby([f'GID_{admin_level}'])['NO2_SUBINDEX'].pct_change()*100
    df_AQI_gb['SO2_PCT_CHANGE'] = df_AQI_gb.groupby([f'GID_{admin_level}'])['SO2_SUBINDEX'].pct_change()*100
    df_AQI_gb['CO_PCT_CHANGE'] = df_AQI_gb.groupby([f'GID_{admin_level}'])['CO_SUBINDEX'].pct_change()*100
    df_AQI_gb['O3_PCT_CHANGE'] = df_AQI_gb.groupby([f'GID_{admin_level}'])['O3_SUBINDEX'].pct_change()*100
    df_AQI_gb = df_AQI_gb.replace([np.inf], 100)
    df_AQI_gb = df_AQI_gb.fillna(0)
    df_AQI_gb = df_AQI_gb[df_AQI_gb['year'].between(2019, 2022)]
    df_AQI_gb.columns = df_AQI_gb.columns.str.upper()

    return df_AQI_gb

def calc_AQI_buckets(df_AQI: pd.DataFrame, admin_level: int) -> pd.DataFrame:
    """
    This function aggregates the number of days the input dataframe's regions have experienced various air quality ratings ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'].
    This is a component to compute the GED Air Quality score 

    Keyword Arguments:
    df_AQI -- AQI pandas dataframe
    admin_level -- integer of the chosen administrative level
    """

    df_AQI = df_AQI[df_AQI['year'] > 2018]

    # Group the data by region, year, and air quality and count the number of days
    df_AQI_bucket = df_AQI.groupby([f'GID_{admin_level}', 'year', 'AQI_bucket'])['date'].size().reset_index(name='count')

    df_AQI_bucket = df_AQI_bucket.pivot_table(index=[f'GID_{admin_level}', 'year'], columns='AQI_bucket', values='count', aggfunc='sum').fillna(0)
    df_AQI_bucket = df_AQI_bucket.divide(df_AQI_bucket.sum(axis=1), axis=0) * 100

    # Reset the index to create separate columns for district, year, and air quality level
    df_AQI_bucket = df_AQI_bucket.reset_index()

    for quality in ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']:
        if quality not in df_AQI_bucket.columns:
            df_AQI_bucket[quality] = 0.0

    return df_AQI_bucket

def calc_temp_percentiles(avg_temp: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the temperature percentiles of the regions within the inputted pandas dataframe.
    This is later used to calculate the number of days these regions have been within these thresholds

    Keyword Argument:
    avg_temp -- average temperature pandas dataframe
    start_date -- string of the starting date of the time window to be used to compute the percentiles
    end_date -- string of the ending date of the time window to be used to compute the percentiles
    admin_level -- integer of the chosen administrative level
    """

    df_2000_2010 = avg_temp[(avg_temp['date'] >= start_date) & (avg_temp['date'] <= end_date)]
    # Calculate the 90th percentile temperature for each region
    temp_90th_percentile = df_2000_2010.groupby(f'GID_{admin_level}')['average_temperature'].quantile(0.9)

    # Calculate the 10th percentile temperature for each region
    temp_10th_percentile = df_2000_2010.groupby(f'GID_{admin_level}')['average_temperature'].quantile(0.1)

    # Combine the two percentiles into a single DataFrame
    temp_percentiles = pd.concat([temp_90th_percentile, temp_10th_percentile], axis=1)
    temp_percentiles.columns = ['temp_90th_percentile', 'temp_10th_percentile']

    return temp_percentiles

def calc_extreme_temp(avg_temp: pd.DataFrame, temp_percentiles: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:
    
    """
    This function calculates the number of days the regions within the inputted dataframe have experienced 'extremely cold' and 'extremely hot' temperatures.
    The output is used to compute the GED's temperature / weather score

    Keyword Arguments:
    avg_temp -- average temperature pandas dataframe
    temp_percentiles -- pandas dataframe containing each region's extreme temperature at the 10th and 90th percentiles from 2000-2010
    start_date -- string of the starting date of the time window to be used to aggregate days a region's experienced extreme temperatures
    end_date -- string of the ending date of the time window to be used to aggredate days a region's experienced extreme temperatures
    admin_level -- integer of the chosen administrative level
    """

    df_extreme_temp_test = avg_temp[(avg_temp['date'] >= start_date) & (avg_temp['date'] <= end_date)]

    # Join the two datasets on the "region" column
    df_extreme_temp_test = pd.merge(df_extreme_temp_test, temp_percentiles, on=f'GID_{admin_level}', how='left')

    # Count the total number of days for each district and year
    total_days = df_extreme_temp_test.groupby([f'GID_{admin_level}', 'year'])['average_temperature'].count()
    total_days = total_days.reset_index()
    total_days = total_days.rename(columns={'average_temperature': 'total_days'})

    # Group the data by district and year, then count the number of days where temp > T_max
    days_above_max = df_extreme_temp_test[df_extreme_temp_test["average_temperature"] > df_extreme_temp_test["temp_90th_percentile"]].groupby([f'GID_{admin_level}', "year"]).size().reset_index(name='extremely_hot_count')
    days_above_max = pd.merge(days_above_max, total_days, on=[f'GID_{admin_level}', 'year'], how='left')
    # Create a new column with the ratio of 'temp' to 'max'
    days_above_max['extremely_hot'] = days_above_max['extremely_hot_count'] / days_above_max['total_days']
    # Select the desired columns and drop any duplicates
    days_above_max= days_above_max[[f'GID_{admin_level}', 'year', 'extremely_hot']].drop_duplicates()

    # Group the data by district and year, then count the number of days where temp < T_min
    days_below_min = df_extreme_temp_test[df_extreme_temp_test["average_temperature"] < df_extreme_temp_test["temp_10th_percentile"]].groupby([f'GID_{admin_level}', "year"]).size().reset_index(name='extremely_cold_count')
    days_below_min = pd.merge(days_below_min, total_days, on=[f'GID_{admin_level}', 'year'])
    # Create a new column with the ratio of 'temp' to 'max'
    days_below_min['extremely_cold'] = days_below_min['extremely_cold_count'] / days_below_min['total_days']
    # Select the desired columns and drop any duplicates
    days_below_min= days_below_min[[f'GID_{admin_level}', 'year', 'extremely_cold']].drop_duplicates()

    # Merge the two results into a single DataFrame
    df_extreme_temp = pd.merge(days_above_max, days_below_min, on=[f'GID_{admin_level}', 'year'], how='outer')
    df_extreme_temp = df_extreme_temp.fillna(0)

    df_extreme_temp = df_extreme_temp.sort_values(by=[f'GID_{admin_level}', 'year']).reset_index(drop=True)
    df_extreme_temp['extremely_hot_pct_change'] = df_extreme_temp.groupby([f'GID_{admin_level}'])['extremely_hot'].pct_change() * 100 
    df_extreme_temp['extremely_cold_pct_change'] = df_extreme_temp.groupby([f'GID_{admin_level}'])['extremely_cold'].pct_change() * 100 
    df_extreme_temp = df_extreme_temp[df_extreme_temp['year'] > 2018]

    return df_extreme_temp

def calc_max_temp(max_temp: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the maximum temperature values of the inputted dataframe's regions. 
    This is used to compute the GED's temperature / weather score
    
    Keyword Arguments:
    max_temp -- maximum temperature pandas dataframe
    start_date -- string of the starting date of the time window to be used to isolate the dataset's desired time window
    end_date -- string of the ending date of the time window to be used to isolate the dataset's desired time window
    admin_level -- integer of the chosen administrative level
    """

    df_max_temp = max_temp[(max_temp['date'] >= start_date) & (max_temp['date'] <= end_date)]

    # Group the data by district and year and add max temperature of each year in each district
    df_max_temp = df_max_temp.groupby([f'GID_{admin_level}', 'year'])['max_temp'].max().reset_index()
    df_max_temp['MAX_TEMP_PCT_CHANGE'] = 100 * df_max_temp.groupby([f'GID_{admin_level}'])['max_temp'].pct_change()
    df_max_temp = df_max_temp[df_max_temp['year'] > 2018]
    
    return df_max_temp

def calc_max_prec(precipitation: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the maximum precipitation values of the inputted dataframe's regions. 
    This is used to compute the GED's temperature / weather score
    
    Keyword Arguments:
    precipitation -- precipitation pandas dataframe
    start_date -- string of the starting date of the time window to be used to isolate the dataset's desired time window
    end_date -- string of the ending date of the time window to be used to isolate the dataset's desired time window
    admin_level -- integer of the chosen administrative level
    """

    df_prec_max = precipitation[(precipitation['date'] >= start_date) & (precipitation['date'] <= end_date)]

    # Group the data by district and year and add max percipitation of each year in each district
    df_prec_max = df_prec_max.groupby(['year', f'GID_{admin_level}'])['precipitation'].max().reset_index()
    df_prec_max = df_prec_max.rename(columns={'precipitation': 'precipitation_max'})
    df_prec_max = df_prec_max.sort_values(by=[f'GID_{admin_level}', 'year']).reset_index(drop=True)
    df_prec_max['precipitation_max_pct_change'] = 100*df_prec_max.groupby([f'GID_{admin_level}'])['precipitation_max'].pct_change()
    df_prec_max = df_prec_max[df_prec_max['year'] > 2018]

    return df_prec_max

def calc_prec_percentiles(precipitation: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the precipitation percentiles of the regions within the inputted pandas dataframe.
    This is later used to calculate the number of days these regions have been within these thresholds

    Keyword Argument:
    avg_temp -- precipitation pandas dataframe
    start_date -- string of the starting date of the time window to be used to compute the percentiles
    end_date -- string of the ending date of the time window to be used to compute the percentiles
    admin_level -- integer of the chosen administrative level
    """

    df_2000_2010 = precipitation[(precipitation['date'] >= start_date) & (precipitation['date'] <= end_date)]

    # Calculate the 90th percentile perc for each region
    P_max = df_2000_2010.groupby(f'GID_{admin_level}')['precipitation'].quantile(0.9)
    # Calculate the 10th percentile perc for each region
    P_min = df_2000_2010.groupby(f'GID_{admin_level}')['precipitation'].quantile(0.1)
    # Combine the two percentiles into a single DataFrame
    prec_percentiles = pd.concat([P_max , P_min], axis=1)
    prec_percentiles.columns = ['P_max', 'P_min']
    
    return prec_percentiles

def calc_extreme_prec(precipitation: pd.DataFrame, prec_percentiles: pd.DataFrame, start_date: str, end_date: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the number of days the regions within the inputted dataframe have experienced 'extremely dry' and 'extremely wet' climates.
    The output is used to compute the GED's temperature / weather score

    Keyword Arguments:
    precipitation -- precipitation pandas dataframe
    temp_percentiles -- pandas dataframe containing each region's extreme precipitation at the 10th and 90th percentiles from 2000-2010
    start_date -- string of the starting date of the time window to be used to aggregate days a region's experienced extreme precipitation
    end_date -- string of the ending date of the time window to be used to aggredate days a region's experienced extreme precipitation
    admin_level -- integer of the chosen administrative level
    """

    df_extreme_prec = precipitation[(precipitation['date'] >= start_date) & (precipitation['date'] <= end_date)]

    # Join the two datasets on the "region" column
    df_extreme_prec = pd.merge(df_extreme_prec, prec_percentiles, on=f'GID_{admin_level}')

    # Count the total number of days for each district and year
    total_days_prec = df_extreme_prec.groupby([f'GID_{admin_level}', 'year'])['precipitation'].count()
    total_days_prec = total_days_prec .reset_index()
    total_days_prec = total_days_prec.rename(columns={'precipitation': 'total_days'})

    # Group the data by district and year, then count the number of days where temp > T_max
    days_above_max = df_extreme_prec[df_extreme_prec["precipitation"] > df_extreme_prec["P_max"]].groupby([f'GID_{admin_level}', "year"]).size().reset_index(name='extremely_wet_count')
    days_above_max = pd.merge(days_above_max, total_days_prec, on=[f'GID_{admin_level}', 'year'])
    # Create a new column with the ratio of 'temp' to 'max'
    days_above_max['extremely_wet'] = days_above_max['extremely_wet_count'] / days_above_max['total_days']
    # Select the desired columns and drop any duplicates
    days_above_max= days_above_max[[f'GID_{admin_level}', 'year', 'extremely_wet']].drop_duplicates()

    # Group the data by district and year, then count the number of days where temp < T_min
    days_below_min = df_extreme_prec[df_extreme_prec["precipitation"] < df_extreme_prec["P_min"]].groupby([f'GID_{admin_level}', "year"]).size().reset_index(name='extremely_dry_count')
    days_below_min = pd.merge(days_below_min, total_days_prec, on=[f'GID_{admin_level}', 'year'])
    # Create a new column with the ratio of 'temp' to 'max'
    days_below_min['extremely_dry'] = days_below_min['extremely_dry_count'] / days_below_min['total_days']
    # Select the desired columns and drop any duplicates
    days_below_min= days_below_min[[f'GID_{admin_level}', 'year', 'extremely_dry']].drop_duplicates()

    # Merge the two results into a single DataFrame
    df_extreme_prec = pd.merge(days_above_max, days_below_min, on=[f'GID_{admin_level}', 'year'], how='outer')
    df_extreme_prec = df_extreme_prec.fillna(0)

    df_extreme_prec = df_extreme_prec.sort_values(by=[f'GID_{admin_level}', 'year']).reset_index(drop=True)
    df_extreme_prec['extremely_wet_pct_change'] = df_extreme_prec.groupby([f'GID_{admin_level}'])['extremely_wet'].pct_change() * 100 
    df_extreme_prec['extremely_dry_pct_change'] = df_extreme_prec.groupby([f'GID_{admin_level}'])['extremely_dry'].pct_change() * 100 
    df_extreme_prec = df_extreme_prec[df_extreme_prec['year'] > 2017]

    df_extreme_prec = df_extreme_prec.replace([np.inf], 100)

    return df_extreme_prec


def calc_lpc(bm: pd.DataFrame, pop: pd.DataFrame, start_year: str, end_year: str, admin_level: int) -> pd.DataFrame:

    """
    This function calculates the country's luminosity per capita using nighttime luminosity and population data and the chosen administrative level.
    This is used to compute the GED's economic score 

    Keyword Arguments:
    bm -- black marble luminosity pandas dataframe
    pop -- population pandas dataframe
    start_date -- string of the starting date of the time window to be used to isolate the dataset's desired time window
    end_date -- string of the ending date of the time window to be used to isolate the dataset's desired time window
    admin_level -- integer of the chosen administrative level
    """

    df_econ = pd.merge(bm,
                       pop,
                       on=[f'GID_{admin_level}', 'year'], how='left')
    
    df_econ = df_econ.fillna(0)

    df_econ['LPC'] = df_econ['luminosity'] / (1+df_econ['population'])

    df_econ = df_econ.sort_values(by=[f'GID_{admin_level}', 'year'], ascending=True)
    df_econ['LPC_PCT_CHANGE'] = \
                    100*(df_econ.groupby(f'GID_{admin_level}')['LPC'].apply(pd.Series.pct_change))
    df_econ = df_econ[df_econ['year'].between(start_year, end_year)].reset_index(drop=True)
    df_econ = df_econ.replace([np.inf], 100)
    df_econ = df_econ[df_econ['year'].between(start_year+1, end_year)]

    return df_econ


def get_pca_comp(df: pd.DataFrame, variables: list) -> pd.DataFrame:

    """
    This function uses Principal Component Analysis to combine desired variables of the inputted dataframe into a single index
    The output is used as a score [air quality, weather, deforestation, economic]

    Keyword Arguments:
    df -- pandas dataframe containing the score components 
    variables -- list of variables to be used to be part of the PCA 
    
    """

    X = df[variables]

    # Scale the data to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use PCA to perform dimensionality reduction
    pca = PCA()
    pca.fit(X_scaled)   

    # Determine how many principal components to keep based on the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    n_components = len([evr for evr in explained_variance_ratio if evr >= 0.05])

    # Fit PCA with the chosen number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate the principal component scores for each observation in the original data
    scores = pd.DataFrame(X_pca, columns=['PC{}'.format(i+1) for i in range(n_components)])
    scores.index = df.index

    # Combine the principal component scores into a single index by assigning weights
    # based on the explained variance ratio of each component
    weights = explained_variance_ratio[:n_components]
    index = scores.dot(weights)

    return index

def calc_z_scores(df: pd.DataFrame, z_cols: list) -> pd.DataFrame:

    """
    This function calculates the z scores / standard deviations of the chosen columns

    Keyword Argument:
    df -- pandas dataframe containing all components required to compute GED scores
    z_cols -- list of desired colummns to be used to compute z scores 

    """
    for col in z_cols:
        df[col+'_STD'] = df[[col]].apply(zscore)
    
    return df