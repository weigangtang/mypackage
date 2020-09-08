# See hydrometric_plotly_trend_map.ipynb for the application

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os, conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap

import plotly.graph_objs as go
import plotly.express as px
mapbox_token = 'pk.eyJ1IjoidzI5dGFuZyIsImEiOiJjazZ4c3R4bGgwcWtyM2ZyeHRtejM3MmEzIn0.V-mxo1kAs_gTn7jPfTy4Yw'
px.set_mapbox_access_token(mapbox_token) 

def binary_classify(x, thr, v1, v2):
    x_class = np.digitize(x, [thr]) * (v2 - v1) + v1
    return x_class

# Color Functions ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# 3 Types of Color Map
# - matplotlib colormap object (Basemap)
# - RGB dict (adopted by Plotly)
# - RGB color vector

# RGB for plotly: 0-255
# RGB for matplotlib / Basemap: 0-1

def get_continuous_colors(cmap_name, nlev, clim=[0, 1]):
    color_map = plt.cm.get_cmap(cmap_name)
    rgba_vec = color_map(np.linspace(clim[0], clim[1], nlev))
    rgb_vec = rgba_vec[:, :3]
    return rgb_vec

def get_discrete_colors(cmap_name):
    color_map = plt.cm.get_cmap(cmap_name)
    rgb_vec = np.array(color_map.colors)
    return rgb_vec

def cut_continuous_color_map(cmap_name, nlev, clim=[0, 1]):
    color_map = plt.cm.get_cmap(cmap_name, 512)
    color_map = ListedColormap(color_map(np.linspace(clim[0], clim[1], nlev)))
    return color_map

def cut_discrete_color_map(cmap_name, nlev):
    color_map = plt.cm.get_cmap(cmap_name)
    color_map = ListedColormap(color_map.colors[:nlev])
    return color_map

# Color Scale for Plotly map ONLY
# a Dictionary - keys: level names; values: rgb string (e.g. rgb(192, 128, 64)) 
def get_continuous_color_scale(cmap_name, nlev, clim=[0, 1]):
    rgb_vec = get_continuous_colors(cmap_name, nlev, clim)
    rgb_vec = (rgb_vec * 255).astype(int)
    rgb_str = ['rgb({},{},{})'.format(*item) for item in rgb_vec]
    levs = np.linspace(0, 1, nlev)
    color_scale = list(zip(levs, rgb_str))
    return color_scale

def get_discrete_color_scale(cmap_name, keys):
    rgb_vec = get_discrete_colors(cmap_name)
    rgb_vec = (rgb_vec * 255).astype(int)
    rgb_str = ['rgb({},{},{})'.format(*item) for item in rgb_vec]
    n_key = len(keys); n_color = len(rgb_str)
    assert n_key <= n_color
    color_scale = dict(zip(keys, rgb_str[:n_key]))
    return color_scale

# Printable Map (Basemap) -------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

map_config_wna = {'lat_0':45, 'lon_0':-108, 
    'llcrnrlon':-135, 'llcrnrlat':24, 'urcrnrlon':-80, 'urcrnrlat':76}

# Example
# color_map = cut_continuous_color_map('RdBu_r', 10, [0.05, 0.95])
# style_config = {'cmap': color_map, 'clim': [-3.5, 3.5]}
def generate_printable_map(df, map_config, style_config):
    
    if 'SIZE' not in df.columns: df['SIZE'] = 1.
    if 'OPACITY' not in df.columns: df['OPACITY'] = 1.
    
    # light theme 
    line_color = 'grey'; land_color = 'white'; water_color = [0.9, 0.9, 0.9]
    
    # dark theme
    # line_color = 'white'; land_color = [0.1, 0.1, 0.1]; water_color = [0.5, 0.5, 0.5]
    
    fig = plt.figure()
    m = Basemap(projection='eqdc', resolution='h', **map_config)
    m.drawcountries(linewidth=0.8, color=line_color)
    m.drawstates(linewidth=0.5, color=line_color, linestyle='--')
    # m.drawcoastlines(linewidth=0.5, color=land_color)
    m.drawmapboundary(fill_color=water_color) # ocean
    m.fillcontinents(color=land_color, lake_color=water_color) # land
    
    x, y = m(df['LONGITUDE'].values, df['LATITUDE'].values)
    s, c = df['SIZE'].values, df['COLOR'].values

    opacity = df['OPACITY'].values
    for opc in np.unique(opacity):
        indx = opacity == opc
        # clim is not avaiable in scatter, using vmin, vmax instead
        ax = m.scatter(x[indx], y[indx], c=c[indx], s=s[indx], alpha=opc, zorder=5)
        if 'cmap' in style_config: ax.set_cmap(style_config['cmap'])
        if 'clim' in style_config: ax.set_clim(style_config['clim']) 
    
    width = 10; height = width * m.aspect
    fig.set_size_inches([width, height])
    
    return fig

# Figure Widget (Plotly) --------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Example:
# color_scale = create_color_scale('RdBu_r', 11, [0.1, 0.9])
# style_config = {'sizemin': 2., 'colorscale': color_scale, 'cmin':-0.5, 'cmax':0.5}

def generate_figure_widget(df, style_config):

    req_cols = ['LATITUDE', 'LONGITUDE', 'SIZE', 'COLOR', 'OPACITY', 'HOVER NAME']
    error_message_frame = '{} Not Found!'
    for colname in req_cols:
        assert colname in df.columns, error_message_frame.format(colname)

    trace = go.Scattermapbox(
        lat = df['LATITUDE'],
        lon = df['LONGITUDE'],
        mode='markers',
        marker = dict(
            color = df['COLOR'], 
            size = df['SIZE'],
            opacity = df['OPACITY'],  
            **style_config, 
        ),
        hovertext =  df['HOVER NAME'],
        selected = dict(marker={'color': 'lime'}), 
    )

    layout = dict(
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_token,
            center=go.layout.mapbox.Center(lat=52.5, lon=-105),
            style='dark',
            zoom=2., 
        ), 
        width = 900, height = 500, margin = {'l':10, 'r':10, 't':10, 'b':10}
    )
    
    fw = go.FigureWidget(data=[trace], layout=layout)
    return fw

# HOVER NAME must exist
def get_select_pts(fw):
    sel_idx = list(fw.data[0]['selectedpoints'])
    all_sid = fw.data[0]['hovertext']
    sel_sid = all_sid[sel_idx].tolist()
    return sel_sid


# Interactive Map (Plotly) ------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Example: 
# color_scale = get_continuous_color_scale('RdBu_r', 11, [0.1, 0.9])
# style_config = {'range_color': [-30., 30.], 'color_continuous_scale': color_scale}

def generate_interactive_map(df, style_config):
    
    cnames = df.columns.tolist()
    if 'SIZE' not in cnames: df.loc[:, 'SIZE'] = 1.
    if 'COLOR' not in cnames: df.loc[:, 'COLOR'] = 1.
    if 'OPACITY' not in cnames: df.loc[:, 'OPACITY'] = 1.
    if 'HOVER NAME' not in cnames: df.loc[:, 'HOVER NAME'] = df.index
    
    fig = px.scatter_mapbox(df, lat='LATITUDE', lon='LONGITUDE', 
                         size='SIZE', color='COLOR', opacity=df['OPACITY'],  
                         hover_name='HOVER NAME', 
                         center = {'lat': 52.5, 'lon': -105}, zoom=2.2,
                         **style_config)
    
    fig.update_layout(mapbox_style='dark', coloraxis_showscale=False)
    return fig


# Printable Geo Map (Plotly) ----------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Example:
# map_config = {'scope': 'north america'
#     'projection': {'type': 'conic equidistant', 'rotation_lon': -108}, 
#     'lataxis': {'range': [28, 75]}, 'lonaxis': {'range': [-143, -103]}}
# 
# color_scale = get_continuous_color_scale('RdBu_r', 50, [0.05, 0.95])
# style_config = {'range_color': [-1., 1.], 'color_continuous_scale': color_scale}

def generate_printable_geomap(df, map_config, style_config):
    
    cnames = df.columns.tolist()
    if 'SIZE' not in cnames: df.loc[:, 'SIZE'] = 5.
    if 'COLOR' not in cnames: df.loc[:, 'COLOR'] = 1.
    if 'OPACITY' not in cnames: df.loc[:, 'OPACITY'] = 1.
    if 'HOVER NAME' not in cnames: df.loc[:, 'HOVER NAME'] = df.index
    
    fig = px.scatter_geo(df, lat='LATITUDE', lon='LONGITUDE', 
        size='SIZE', color='COLOR', opacity=df['OPACITY'], hover_name='HOVER NAME',
        **style_config)
    fig.update_traces(marker = {'line': {'width': 0}})
    
    landcolor = 'rgb(80, 80, 80)'
    lakecolor = 'rgb(32, 32, 32)'
    fig.update_geos(
        resolution = 50, 
        showland = True, landcolor =  landcolor,
        showlakes = True, lakecolor = lakecolor, 
        showocean = True, oceancolor = lakecolor, 
        showcoastlines = True, coastlinecolor = lakecolor, coastlinewidth = .8, 
        showcountries = True, countrycolor = 'white', countrywidth = .5, 
        showsubunits  = True, subunitcolor = 'white', subunitwidth = .5, 
        **map_config,
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False,)
    return fig

