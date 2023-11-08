#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:10:10 2023

@author: rantanem
"""



import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='wrap')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='wrap')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxx[i]], lat[mxy[i]], symbol, color=color, size=28, fontweight = 'bold',
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                transform=transform, zorder=14)
        ax.text(lon[mxx[i]], lat[mxy[i]],
                '\n' + str(int(data[mxy[i], mxx[i]])),
                color=color, size=18, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', 
                transform=transform, zorder=14)

def plotMap():
    
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import matplotlib.ticker as mticker


    #Set the projection information
    proj = ccrs.LambertConformal(central_longitude=20.0,central_latitude=62, standard_parallels=[62])
    # proj = ccrs.PlateCarree()
    #Create a figure with an axes object on which we will plot. Pass the projection to that axes.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8),subplot_kw=dict(projection=proj),dpi=300)
    
    #Zoom in
    ax.set_extent([38, 2, 52, 72])
    
    #Add map features
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.95') #Grayscale colors can be set using 0 (black) to 1 (white)
 #   ax.add_feature(cfeature.LAKES, alpha=0.9)  #Alpha sets transparency (0 is transparent, 1 is solid)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), zorder=8,linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), zorder=8,linewidth=0.5)

    #We can use additional features from Natural Earth (http://www.naturalearthdata.com/features/)
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',  name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray', zorder=10)
    
    #Add lat/lon gridlines every 20° to the map
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    gl.right_labels = gl.top_labels = False
    gl.ylocator = mticker.FixedLocator(np.arange(40,75,5))
    gl.xlocator = mticker.FixedLocator(np.arange(0,60,10))
    gl.ylabel_style = {'size': 14, 'color': 'k'}
    gl.xlabel_style = {'size': 14, 'color': 'k'}
    
    return fig, ax

def draw_circle(ax, lon, lat, color, radius):

    import shapely.geometry as sgeom
    from cartopy.geodesic import Geodesic
    gd = Geodesic()
    
    geoms = []
    cp = gd.circle(lon=lon, lat=lat, radius=700000.)
    geoms.append(sgeom.Polygon(cp))

    # Note the specification of coordinate transformation, using the
    # .. correct parameter:  crs=src_crs
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), edgecolor=color, linewidth=1.7,
                      facecolor='None',zorder=10)
    
    return ax
                      
                
input_path = '/Users/rantanem/Documents/python/mawecli/scc_dates/'
   
    

mareographs = {'kemi' :        [65.67, 24.52],
               'helsinki':     [60.15, 24.96],
               'pärnu':        [58.382, 24.477],
               'landsort':    [58.75, 17.8667],
               # 'kungholmsfort':[56.11, 15.59],
               }
colors = {'kemi' :   'orange',
          'helsinki':'blue',
          'landsort':'lime',
          'pärnu': 'red',

          }




da = xr.open_dataarray('/Users/rantanem/Documents/python/mawecli/data/era5_mslp_jan2005.nc')/100
f = da.sel(time='2005-01-09T00:00')
palette='RdBu_r'
f_levels = np.arange(960,1025,5)

# mark certain maregraphs red
markred=False

#Get a new background map figure
fig, ax = plotMap()

f_contourf = ax.contourf(f.longitude, f.latitude, f, levels = f_levels, zorder=2,  
                         cmap=palette, transform = ccrs.PlateCarree(), 
                         extend='both', alpha=0.85)

for key in mareographs.keys():
    if key[0]=='h':
        rot = 45
    else:
        rot=0
    if key[0]=='f':
        y=-0.35;x=-0.02
    elif key[0]=='l':
        x=-5.5;y=0.1
    else:
        y=0.2;x=0.2
        
        
    plt.scatter(mareographs[key][1], mareographs[key][0],s=130,
          color='k',marker="^",
          transform=ccrs.PlateCarree(),zorder=10
          )
    plt.text(mareographs[key][1]+x, mareographs[key][0]+y,
          key.capitalize(),fontsize=14,color='k',rotation=rot,
          transform=ccrs.PlateCarree(),zorder=10
          )

etc_tracks  = pd.read_csv(input_path +'pärnu'+'_close_cyclones.csv', index_col=0, parse_dates=True)
etc_tracks = etc_tracks.loc[slice('2005-01-01','2005-01-10')]

track_ids = etc_tracks.track_id.unique()

col = ['orange', 'blue','k','lime','coral']

for i, track in enumerate(track_ids[:-1]):
    track = etc_tracks[etc_tracks.track_id == track]
    
    ax.plot(track.lon_centre, track.lat_centre, '-o', linewidth=1, markersize=6, alpha=1, zorder=10,
            transform = ccrs.PlateCarree(), color=col[i])


track = etc_tracks[etc_tracks.track_id == etc_tracks.track_id.unique()[2]]

ax.plot(track.lon_centre, track.lat_centre, '-o', linewidth=1, markersize=6, alpha=1, zorder=10,
        transform = ccrs.PlateCarree(), color='red')

# plot_maxmin_points(ax, f.longitude, f.latitude, f, 'min', 400, 'L', color='red',
#                        plotValue=True, transform = ccrs.PlateCarree())  
### mark certain tide gauges red

if markred:
    for tg in ['kemi','rauma','helsinki','hamina']:
    
        plt.scatter(mareographs[tg][1], mareographs[tg][0],s=100,
                    color='r',marker="^",zorder=10,
                    transform=ccrs.PlateCarree(),
                    )
 
for key in ['kemi']:
    lon = mareographs[key][1]
    lat = mareographs[key][0]
    draw_circle(ax, lon, lat, color='k', radius=700)
    
ax.text(22, 64.2,'2.1.05', color='orange', fontsize=16,fontweight='bold',
        transform=ccrs.PlateCarree(),zorder=10)
ax.text(35, 57.5,'5.1.05', color='blue', fontsize=16,fontweight='bold',
        transform=ccrs.PlateCarree(),zorder=10)
ax.text(28.3, 63,'7.1.05', color='lime', fontsize=16,fontweight='bold',
        transform=ccrs.PlateCarree(), zorder=10)
ax.text(33, 59.5,'8.1.05\n(Gudrun)', color='red', fontsize=16,fontweight='bold',
        transform=ccrs.PlateCarree(),zorder=10)
    
# add colorbar and its specifications
fig.subplots_adjust(wspace=0.07)
cbar_ax = fig.add_axes([0.92, 0.17, 0.023, 0.65])
cb = fig.colorbar(f_contourf, orientation='vertical',pad=0.05,
                  fraction=0.053, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
cb.set_label(label='Sea level pressure [hPa]',fontsize=16)
    
# plt.text(23.8, 59.56,
#          "Gulf of Finland",fontsize=14,color='k',rotation=16,fontstyle='italic',
#          transform=ccrs.PlateCarree(),zorder=10)

# plt.text(19.1, 60.9,
#          "Bothnian Sea",fontsize=14,color='k',rotation=80,fontstyle='italic',
#          transform=ccrs.PlateCarree(),zorder=10)

# plt.text(21.5, 63.7,
#          "Bothnian Bay",fontsize=14,color='k',rotation=55,fontstyle='italic',
#          transform=ccrs.PlateCarree(),zorder=10)

ax.text(18.2, 56.5,
          "Baltic Sea",fontsize=14,color='k',rotation=58,fontstyle='italic',
          transform=ccrs.PlateCarree(),zorder=10)



# # save figure
figurePath = '/Users/rantanem/Documents/python/mawecli/figures/'
figureName =  'fig01.pdf'
   
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')

figureName =  'fig01.png'
   
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')

