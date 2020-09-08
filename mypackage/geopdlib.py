import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import glob

import geopandas as gpd
import fiona, json, shapely

import multiprocessing as mp
from functools import partial

def get_file_size(fpath):
    fsize = os.path.getsize(fpath)
    fsize = np.round(fsize / 1e6, 2)
    return fsize

def get_json_size(json_str):
    return np.round(len(json_str)/1e6, 2)

# Projection
wgs84 = {'init': 'epsg:4326'}
wgs84_mercator = {'init': 'epsg:3857'}
utm_north = {'init': 'epsg:32633'}
utm_south = {'init': 'epsg:32733'}
# Eckert VI (equal-area projection) Preserve Area (in sqkm)
eckert6_prj4 = '+proj=eck6 +lon_0=0 +x_0=0 +y_0=0 +a=6371000 +b=6371000 +units=m +no_defs'
def calculate_area(plys, prj=eckert6_prj4):
    return plys.to_crs(prj).area / 1e6

def calculate_area_mp(plys, prj=eckert6_prj4, chunk_size=100):
    n = len(plys)
    if n <= chunk_size * 2:
        print('Single Core Processing ...')
        plys_area = calculate_area(plys, prj=prj)
    else:
        print('Multi Core Porcessing ....')
        pool = mp.Pool(processes=8)
        plys_list = [plys[i:i+chunk_size] for i in range(0, n, chunk_size)]
        area_list = pool.map(partial(calculate_area, prj=prj), plys_list)
        plys_area = pd.concat(area_list)
        pool.close() # release memory
    return plys_area


def simplify(plys, tr):
    return plys.simplify(tolerance=tr, preserve_topology=True)
    
def simplify_mp(plys, tr, chunk_size):
    n = len(plys)
    if n <= chunk_size * 2:
        print('Single Core Processing ...')
        plys_simp = simplify(plysm, tr=tr)
    else:
        print('Multi Core Porcessing ....')
        pool = mp.Pool(processes=8)
        plys_list = [plys[i:i+chunk_size] for i in range(0, n, chunk_size)]
        simp_list = pool.map(partial(simplify, tr=tr), plys_list)
        plys_simp = pd.concat(simp_list)
        pool.close() # release memory
    return plys_simp


# remove elevation (3D to 2D)
# round digit to 6
# add id key, gdf must has a column named <ID>
def process_json(myjson):
    features = []
    for item in myjson['features']:
        crds = item['geometry']['coordinates'][0]
        crds = [np.round(crd[:2], 6).tolist() for crd in crds]
        item['geometry']['coordinates'][0] = crds
        item['id'] = item['properties']['ID']
        features.append(item)
    myjson['features'] = features
    return myjson

def break_multi_polygons(gdf_input):
    prj = gdf_input.crs
    plys = []
    for item in gdf_input['geometry']:
        if isinstance(item, shapely.geometry.polygon.Polygon):
            plys += [item]
        else:
            plys += list(item)
    gdf_output = gpd.GeoDataFrame(geometry = gpd.GeoSeries(plys))
    gdf_output.crs = prj
    return gdf_output

def rename_shapefile(folder, shp_name_1, shp_name_2):
    file_list = glob.glob(folder + shp_name_1 + '.*')
    fname_list = [item.split('.')[-1] for item in file_list]
    print('Found {} files'.format(len(file_list)))
    for ext in fext_list:
        fpath_1 = folder + shp_name_1 + '.' + ext
        fpath_2 = folder + shp_name_2 + '.' + ext
        os.rename(fpath_1, fpath_2)

def count_nodes(polygon):
    return len(list(polygon.exterior.coords))

# copy from github
# https://gist.github.com/rmania/8c88377a5c902dfbc134795a7af538d8
from shapely.geometry import Polygon, MultiPolygon, shape, Point
def convert_3d_to_2d(geometry):
    '''
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    '''
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo