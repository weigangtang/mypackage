import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib 

import plotly.graph_objs as go
import plotly.express as px
mapbox_token = 'pk.eyJ1IjoidzI5dGFuZyIsImEiOiJjazZ4c3R4bGgwcWtyM2ZyeHRtejM3MmEzIn0.V-mxo1kAs_gTn7jPfTy4Yw'
px.set_mapbox_access_token(mapbox_token) 


def binary_classify(x, thr, v1, v2):
    x_class = np.digitize(x, [thr]) * (v2 - v1) + v1
    return x_class

# color_scale: a list of RGB vectors
# color_dict:  key as name, values as RGB vector

# RGB for plotly: 0-255
# RGB for matplotlib / Basemap: 0-1

# Colormap for plotly must attached with name/value of levels


# Example: create_color_scale('seismic', 11, [0.2, 0.8])
def create_color_scale(color_map_name, n_level, clim): 

    cmap = matplotlib.cm.get_cmap(color_map_name)
    rgba = cmap(np.linspace(clim[0], clim[1], n_level))
    rgb = (rgba[:, :3] * 255).astype(int)
    clevels = ['rgb({}, {}, {})'.format(*rgb[i, :]) for i in range(n_level)]
   
    colorscale = list(zip(np.linspace(0, 1, n_level), clevels))
    return colorscale

# Example: create_discrete_colormap(grp_names, 'Set1')
def create_discrete_colormap(keys, color_map_name):
    colors = eval('px.colors.qualitative.' + color_map_name)
    assert len(keys) <= len(colors)
    color_map = {}
    for i, key in enumerate(keys): 
        color_map[key] = colors[i]
    return color_map

color_config = {'range_color': [-3.5, 3.5], 'color_continuous_scale': 'RdBu_r'}

# Interactive Map ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

mapbox_config_na = {'center': {'lat': 52.5, 'lon': -105}, 'zoom': 2.2}

def generate_interactive_map(df, map_config, style_config):
    
    cnames = df.columns.tolist()
    if 'SIZE' not in cnames: df.loc[:, 'SIZE'] = 1.
    if 'COLOR' not in cnames: df.loc[:, 'COLOR'] = 1.
    if 'OPACITY' not in cnames: df.loc[:, 'OPACITY'] = 1.
    if 'HOVER NAME' not in cnames: df.loc[:, 'HOVER NAME'] = df.index
    
    fig = px.scatter_mapbox(df, lat='LATITUDE', lon='LONGITUDE', 
                         size='SIZE', color='COLOR', opacity=df['OPACITY'],  
                         hover_name='HOVER NAME', 
                         **style_config, **map_config)
    
    fig.update_layout(mapbox_style='dark', coloraxis_showscale=False)
    return fig


# Printable Map -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# recommended setting: 
# size_min, size_max = 8, 80
# margin = {'r': 20, 't': 20, 'l': 20, 'b': 20}
# width = 3000, height = 2600 (North America)
# width = 2500, height = 3500 (Western North America)

geo_config_na = {
    'projection': {'type': 'conic equidistant', 'rotation_lon': -100}, 
    'lataxis': {'range': [27, 76]}, 'lonaxis': {'range': [-150, -70]},
}

geo_config_wna = {
    'projection': {'type': 'conic equidistant', 'rotation_lon': -108}, 
    'lataxis': {'range': [28, 75]}, 'lonaxis': {'range': [-143, -103]},
    'scope': 'north america', 
    'width': 2500, 'height': 3500,
}

def generate_printable_geomap(df, space_config, color_config):
    
    cnames = df.columns.tolist()
    if 'SIZE' not in cnames: df.loc[:, 'SIZE'] = 5.
    if 'COLOR' not in cnames: df.loc[:, 'COLOR'] = 1.
    if 'OPACITY' not in cnames: df.loc[:, 'OPACITY'] = 1.
    if 'HOVER NAME' not in cnames: df.loc[:, 'HOVER NAME'] = df.index
    
    fig = px.scatter_geo(df, lat='LATITUDE', lon='LONGITUDE', 
                         size='SIZE', color='COLOR', hover_name='HOVER NAME', opacity=df['OPACITY'],  
                         **color_config)
    fig.update_traces(marker = {'line': {'width': 0}})
    
    landcolor = 'rgb(80, 80, 80)'
    lakecolor = 'rgb(32, 32, 32)'
    fig.update_geos(
        **space_config, resolution = 50, 
        showland = True, landcolor =  landcolor,
        showlakes = True, lakecolor = lakecolor, 
        showocean = True, oceancolor = lakecolor, 
        showcoastlines = True, coastlinecolor = lakecolor, coastlinewidth = .8, 
        showcountries = True, countrycolor = 'white', countrywidth = .5, 
        showsubunits  = True, subunitcolor = 'white', subunitwidth = .5, 
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False,)
    return fig


# Figure Widget -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Example:
# color_scale = create_color_scale('RdBu_r', 11, [0.1, 0.9])
# style_config = {'sizemin':2., 'colorscale': color_scale,  'cmin':-.5, 'cmax':.5, 'opacity': 1.}

def generate_figure_widget(df, style_config):

    req_cols = ['LATITUDE', 'LONGITUDE', 'SIZE', 'COLOR', 'OPACITY', 'HOVER NAME']
    error_message_frame = '{} is missed from input dataframe.'
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