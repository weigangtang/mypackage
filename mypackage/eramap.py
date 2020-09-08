import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

import copy

from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc

# from IPython.display import HTML
# HTML(anim.to_html5_video())

map_frame_global = Basemap(projection='robin', resolution='l', lat_0=0, lon_0=0) # moll, robin, cyl
map_frame_na = Basemap(projection='lcc', resolution='i', width=8e6, height=8e6, lat_0=55, lon_0=-100)

color_config_temp = {'cmap': 'RdBu_r', 'clim': [-40, 40], 'cticks': np.arange(-40, 41, 10)}
color_config_precip = {'cmap': 'Blues', 'clim': [0, 30], 'cticks': np.arange(0, 31, 5)}
color_config_snowfall = {'cmap': 'Blues', 'clim': [0, 10], 'cticks': np.arange(0, 11, 1)}

def plot_map_fast(z, cmap='RdBu_r', clim=None, cticks=None):

    fig = plt.figure(figsize=(13, 6), edgecolor='w'); ax = fig.gca()

    im = ax.imshow(z.astype(float), cmap=cmap)
    if clim is not None:
        im.set_clim(clim)
    if cticks is not None:
        cbar = fig.colorbar(im, ticks=cticks)
        cbar.ax.tick_params(labelsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    
    return fig

def plot_map(lats, lons, z, map_frame, cmap='RdBu_r', clim=None, cticks=None):

    map_frame = copy.copy(map_frame)

    fig = plt.figure(figsize=(13, 6), edgecolor='w'); ax = fig.gca()

    map_frame.drawcoastlines(linewidth=0.5, color='grey')

    x, y = np.meshgrid(lons, lats)
    xi, yi = map_frame(x, y)
    
    quad = map_frame.pcolormesh(xi, yi, z, cmap=cmap)
    if clim is not None:
        quad.set_clim(clim)
    if cticks is not None:
        cbar = fig.colorbar(quad, ticks=cticks)
        cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    
    return fig

def animate_map_fast(dt, z_tensor, cmap='RdBu_r', clim=None, cticks=None):
    
    fig = plt.figure(figsize=(13, 6), edgecolor='w'); ax = fig.gca()

    im = ax.imshow(z_tensor[:, :, 0].astype(float), cmap=cmap)
    if clim is not None:
        im.set_clim(clim)
    if cticks is not None:
        cbar = fig.colorbar(im, ticks=cticks)
        cbar.ax.tick_params(labelsize=12)
    ax.set_title(str(dt[0]), fontsize=24, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]) 
    fig.tight_layout()

    def animate(i):
        im.set_data(z_tensor[:, :, i].astype(float))
        ax.set_title(str(dt[i]), fontsize=24, fontweight='bold')
        return im, 

    anim = animation.FuncAnimation(fig, animate, frames=len(dt), interval=500, blit=True)
    
    return anim


def animate_map(lats, lons, dt, z_tensor, map_frame, cmap='RdBu_r', clim=None, cticks=None, figsize=[13, 6]):

    map_frame = copy.copy(map_frame)
    
    fig = plt.figure(figsize=figsize, edgecolor='w'); ax = fig.gca()
    
    map_frame.drawcoastlines(linewidth=0.5, color='grey')

    x, y = np.meshgrid(lons, lats)
    xi, yi = map_frame(x, y)
    
    z_tensor = np.ma.masked_where(np.isnan(z_tensor), z_tensor)
    quad = map_frame.pcolormesh(xi, yi, z_tensor[:, :, 0], cmap=cmap, shading='gouraud') # can be replaced with ax.pcolormesh 
    quad.cmap.set_bad('white', 1.)
    if clim is not None:
        quad.set_clim(clim)
    if cticks is not None:
        cbar = fig.colorbar(quad, ax=ax, ticks=cticks)
        cbar.ax.tick_params(labelsize=14)
    ax.set_title(str(dt[0]), fontsize=24, fontweight='bold')
    fig.tight_layout()

    def animate(i):
        quad.set_array(z_tensor[:, :, i].ravel())
        ax.set_title(str(dt[i]), fontsize=24, fontweight='bold')
        return quad, 

    anim = animation.FuncAnimation(fig, animate, frames=len(dt), interval=200, blit=True)
    
    return anim
