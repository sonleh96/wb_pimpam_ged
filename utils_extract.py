import os
import requests
import json

import ee
import numpy as np
import xarray as xr
import rioxarray as rxr
import geemap
import geedim
from osgeo import gdal

from dhelper import get_month_range, get_daterange


class Extractor:
    """
    Extractor object for the GED. Contains all methods to extract and pre-process input data
    
    Keyword arguments:
    root -- string parent directory path
    configs -- dictionary containing all user-defined configs and settings
    """
    def __init__(self, root: str, configs: list) -> None:

        # Initiate class attributes 

        self.root = root
        self.configs = configs
        
        # Prepare required directories in which to store raw data 
        try:
            self._prepare_dir()

            self._get_country_shapefiles(admin_level=0)
            self._get_country_shapefiles(admin_level=1)
            self._get_country_shapefiles(admin_level=2)

        except Exception as e:
            print(e)

        # Authenticate into Google Earth Engine and Google Cloud Storage 
        self._authen_ee()
        self._authen_gcs()

        # Create country shapefiles 
        self.region = self._create_region_geometry()

    def _authen_gcs(self):
        """
        Internal class method to authenticate to Google Cloud Storage
        """
        gdal.SetConfigOption('GOOGLE_APPLICATION_CREDENTIALS', 
                             os.path.join(self.root, self.configs['general']['gcs_sv_acc']))
        
        return 
    
    def _authen_ee(self):
        """
        Internal class method to authenticate to Google Earth Engine  
        """
        try:
            ee.Initialize()
        except Exception as e:
            ee.Authenticate()
            ee.Initialize()

        geedim.Initialize()

        return 
    
    def _get_country_shapefiles(self, admin_level: int, version: int=4.1)->dict:
        """
        Internal class method to download selected country's shapefiles from GADM and store it in the appropriate directory

        Keyword arguments:
        admin_level -- integer of the desired administrative level 
        version -- float of the desired GADM data version 
        """
        print(f"Getting {self.configs['general']['country']['name']} shapefiles for admin {admin_level}")

        country_name = self.configs["general"]["country"]["name"]

        out_path = os.path.join(self.root, f'{country_name}/shapefiles/gadm{str(version).replace(".", "")}_{country_name}_{admin_level}.json')

        if not os.path.exists(out_path):

            filename = f'gadm{str(version).replace(".", "")}_{country_name}_{admin_level}.json'
            url = f'https://geodata.ucdavis.edu/gadm/gadm{version}/json/{filename}'
            try:
                r = requests.get(url).json()
                
                with open(f'{out_path}', 'w+') as outfile:
                    json.dump(r, outfile)
            except Exception as e:
                print(e)
        
        return 

    def _prepare_dir(self):
        """
        Internal class method to create the necessary directories to store all data relevant to the chosen country
        """
        if not os.path.exists(os.path.join(self.root, self.configs['general']['country']['name'])):
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name']))
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'shapefiles'))
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data'))
            for c in self.configs['general']['raw_data']['all_components']:
                os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', c))

            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'processed_data'))
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'processed_data', 'admin_1'))
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'processed_data', 'admin_2'))
            os.mkdir(os.path.join(self.root, self.configs['general']['country']['name'], 'datasets'))


    def _create_region_geometry(self):
        """
        Internal class method to create an Earth Engine Geometry
        """
        country_name = self.configs["general"]["country"]["name"]
        path = os.path.join(self.root, f"{country_name}/shapefiles/{self.configs['general']['country']['shapefiles']['lvl_0'].format(country_name)}")
        return geemap.geojson_to_ee(json.load(open(path))).geometry()
    
    def extract_from_nc(self, metric: str) -> None:
        """
        This method extracts raw temperature data from a SISTEMA dataset in netCDF format and stores it in the appropriate directory

        Keyword arguments:
        metric -- string of chosen metric
        """
        ds = xr.open_dataset(self.configs['external'][metric]['source'], decode_coords=True)

        dates_f_half = get_daterange(self.configs['external'][metric]['start_date'][0],
                                     self.configs['external'][metric]['end_date'][0],
                                     input_format="%Y-%m-%d",
                                     output_format="%Y-%m-%d")
        dates_s_half = get_daterange(self.configs['external'][metric]['start_date'][1],
                                     self.configs['external'][metric]['end_date'][1],
                                     input_format="%Y-%m-%d",
                                     output_format="%Y-%m-%d")
        
        for half in [dates_f_half, dates_s_half]:
            for date in half:
                for band in self.configs['external'][metric]['bands']:
                    if band == 'tas':
                        out_path = os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', 'avg_temp', f'{self.configs["general"]["country"]["name"]}_avg_temp_{date}.tif')
                        ds.sel(time=np.datetime64(date))[band].rio.to_raster(out_path, 
                                                                             engine='GeoTIFF')
                    if band == 'tasmax':
                        out_path = os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', 'max_temp', f'{self.configs["general"]["country"]["name"]}_max_temp_{date}.tif')
                        ds.sel(time=np.datetime64(date))[band].rio.to_raster(out_path, 
                                                                             engine='GeoTIFF')
                        
        return 


    def extract_from_gee(self, metric: str, mode: str) -> None:
        """
        Class method to extract a chosen metric from Google Earth Engine and stores it in the appropriate directory

        Keyword arguments:
        metric -- string of chosen metric ['no2', 'so2', ...]
        mode -- string of chosen mode ['mean', 'max', ...]
        """
        if metric not in \
            self.configs['general']['raw_data']['temp_components'] + self.configs['general']['raw_data']['air_components']:
            raise ValueError('Not a valid metric')
        
        if not metric not in ['sum', 'mean', 'max']:
            raise ValueError('Not a valid mode')
        
        dates = get_daterange(self.configs['gee'][metric]['start_date'], 
                              self.configs['gee'][metric]['end_date'], 
                              input_format='%Y-%m-%d', output_format='%Y-%m-%d') 
        
        
        for date in dates:
            dataset = ee.ImageCollection(self.configs['gee'][metric]['source'])\
                .filterDate(f"{date}T00:00:00", f"{date}T23:59:59").select(self.configs['gee'][metric]['bands'])

            if mode == 'sum':
                img = dataset.sum()
            if mode == 'mean': 
                img = dataset.mean()
            if mode == 'max':
                img = dataset.max()

            im = geedim.MaskedImage(img)
            im.download(os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', metric, f'{self.configs["general"]["country"]["name"]}_{metric}_{date}.tif'), 
                        region=self.region, 
                        scale=self.configs['gee'][metric]['scale'], 
                        crs=self.configs['general']['crs'])
            
        return 
    
    def extract_land_cover_from_gee(self):

        """
        Class method to extract land cover raw data from Earth Engine 
        """

        for year in range(self.configs['gee']['land_cover']['start_year'], 
                          self.configs['gee']['land_cover']['end_year']):
            
            dataset = ee.ImageCollection(self.configs['gee']['land_cover']['source'])\
              .filterDate(f"{year}-01-01", f"{year}-12-31").select(self.configs['gee']['land_cover']['bands'])
            
            img = dataset.mode()

            im = geedim.MaskedImage(img)
            im.download(os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', 'land_cover', f'{self.configs["general"]["country"]["name"]}_land_cover_{year}.tif'), 
                        region=self.region, 
                        scale=self.configs['gee']['land_cover']['scale'], 
                        crs=self.configs['general']['crs'])
            
        return 

    def extract_bm(self):
        """
        Class method to extract Black Marble luminosity from Google Cloud Storage 
        """
        yms = get_month_range(self.configs['general']['start_date'][:7], 
                              self.configs['general']['end_date'][:7], 
                              input_format='%Y-%m', output_format='%Y_%m')

        for ym in yms:
            print(ym)
            merged = gdal.Warp(os.path.join(self.root, self.configs['general']['country']['name'], 'raw_data', 'bm', f'{self.configs["general"]["country"]["name"]}_bm_{ym.replace("_", "-")}'),
                               [tile_path.format(ym[:4], ym) for tile_path in self.configs['general']['country']['bm_tiles']],
                                cutlineDSName=self.configs['general']['country']['shapefiles']['lvl_0'],
                                cropToCutline=True)

            merged = None

    

    