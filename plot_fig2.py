
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:30:31 2023

@author: rantanem
"""

import cluster_utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import xarray as xr

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

def plotMap(ax):
    
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import matplotlib.ticker as mticker


    #Set the projection information
    # proj = ccrs.LambertConformal(central_longitude=20.0,central_latitude=62, standard_parallels=[62])
    # proj = ccrs.PlateCarree()
    #Create a figure with an axes object on which we will plot. Pass the projection to that axes.
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8),subplot_kw=dict(projection=proj),dpi=300)
    
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
    
    return 


mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["orange", "blue","lime","red"]) 

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        

### this script reads cyclone track data and calculates various sea level quantities

# define tide gauge coordinates

mareographs = ['kemi', 'helsinki', 'landsort','pärnu']


# input path of the tracks
input_path_sl = '/Users/rantanem/Documents/python/mawecli/Sealevel_Finland_hourlystats_N2000/'
# input path of the clustering data
input_path = '/Users/rantanem/Documents/python/mawecli/scc_dates/'


years = np.arange(1980,2020)

d1 = '2004-11-30'
d2 = '2005-02-01'

sea_level_df = pd.DataFrame(index=pd.date_range(d1,d2, freq='1H'), columns=mareographs)
wb_df = pd.DataFrame(index=pd.date_range(d1,d2, freq='1H'), columns=mareographs)
etc_df = pd.DataFrame(index=pd.date_range(d1,d2), columns=mareographs)
scc3_onset_df = {}
scc2_onset_df = {}
scc1_onset_df = {}
tracks_df = {}

sl_l = {}




for tg in mareographs[:]:
       
    # read sea level and the event onset dates
    df = utils.read_events(tg, input_path_sl, func='all').Sea_level
    wb = df.rolling(window=8*24, center=True).mean()
   
    
    sl_event = pd.read_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_sl_onset_dates.csv', 
                           index_col=0, parse_dates=True)
    sl_event = sl_event.loc[slice(d1,d2)]
    
    
    # read SCC3 & SCC2 onset dates
    scc3_onset_dates  = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True)
    scc3_onset_dates = scc3_onset_dates.loc[slice(d1,d2)]
    scc3_onset_df[tg] = scc3_onset_dates
    
    scc2_onset_dates  = pd.read_csv(input_path +tg+'_scc2.csv', index_col=0, parse_dates=True)
    scc2_onset_dates = scc2_onset_dates.loc[slice(d1,d2)]
    scc2_onset_df[tg] = scc2_onset_dates
    
    scc1_onset_dates  = pd.read_csv(input_path +tg+'_scc1.csv', index_col=0, parse_dates=True)
    scc1_onset_dates = scc1_onset_dates.loc[slice(d1,d2)]
    scc1_onset_df[tg] = scc1_onset_dates
    

    sea_level_df[tg] = df[slice(d1,d2)]
    wb_df[tg] = wb[slice(d1,d2)]
    
    sl_l[tg] = sl_event
    
    scc  = pd.read_csv(input_path +tg+'_scc_statistics.csv', index_col=0, parse_dates=True)
    etc_df[tg] = scc.cyclone_day.loc[slice(d1,d2)]
    
    etc_tracks  = pd.read_csv(input_path +tg+'_close_cyclones.csv', index_col=0, parse_dates=True)
    etc_tracks = etc_tracks.loc[slice('2005-01-01','2005-01-10')]
    tracks_df[tg] = etc_tracks





figure_mosaic = """
A
A
A
A
A
A
A
A
A
A
B
B
B
B
B
C
D
E
F
"""
fig,axarr = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(9,8),dpi=300,sharex=True )
axlist = list(axarr.values())[:]
plt.subplots_adjust(hspace=0)
 
axlist[0].grid(axis='both')


ones = np.ones(len(etc_df))

for i, tg in enumerate(mareographs[:]):
    
    sl_len = sl_l[tg]
    
    a = axlist[0].plot(sea_level_df[tg], linestyle='--',label=tg.capitalize(), linewidth=1 )
    b = axlist[1].plot(wb_df[tg], linestyle='--', linewidth=3,
                       color=a[0].get_color())
    
    
    
    
    for sl_periods in sl_len.index:
        sl_range=pd.date_range(sl_periods, periods=sl_len.loc[sl_periods].length, freq='D')
        axlist[0].plot(sea_level_df[tg].loc[slice(sl_range[0],sl_range[-1])],linewidth=2.5, color=a[0].get_color() )
    
    for scc_periods in scc3_onset_df[tg]['duration'].index:
        axlist[i+2].axvspan(scc_periods, scc_periods + pd.Timedelta(int(scc3_onset_df[tg]['duration'][scc_periods])-1,unit='D'), 
                            alpha=1, color='lightpink',)
        
    for scc_periods in scc2_onset_df[tg]['duration'].index:
        axlist[i+2].axvspan(scc_periods, scc_periods + pd.Timedelta(int(scc2_onset_df[tg]['duration'][scc_periods])-1,unit='D'), 
                            alpha=1, color='lightblue')
        
    for scc_periods in scc1_onset_df[tg]['duration'].index:
        axlist[i+2].axvspan(scc_periods, scc_periods + pd.Timedelta(int(scc1_onset_df[tg]['duration'][scc_periods])-1,unit='D'), 
                            alpha=1, color='lightgrey',)
    
    axlist[i+2].scatter(etc_df[tg][etc_df[tg]==1].index, ones[etc_df[tg]==1], marker=".", s=50, facecolor='k', )
    axlist[i+2].scatter(etc_df[tg][etc_df[tg]==2].index, ones[etc_df[tg]==2], marker="o", s=50, facecolor='k', )


    axlist[i+2].set_xlim(pd.to_datetime(d1)+pd.Timedelta('12 hours'),pd.to_datetime(d2)-pd.Timedelta('12 hours'))
    axlist[i+2].set_yticks([1],labels=[tg.capitalize()], fontsize=12)
    myFmt = mdates.DateFormatter('%d\n%b\n%Y')
    axlist[i+2].xaxis.set_major_formatter(myFmt)

scc3_patch = mpatches.Patch(color='lightpink', label='SCC3')
scc2_patch = mpatches.Patch(color='lightblue', label='SCC2')
scc1_patch = mpatches.Patch(color='lightgrey', label='ETC1')
point1 = Line2D([0], [0], label='1 cyclone', marker='.', markersize=8, 
         markeredgecolor='k', markerfacecolor='k', linestyle='')
point2 = Line2D([0], [0], label='2 cyclones', marker='o', markersize=8, 
         markeredgecolor='k', markerfacecolor='k', linestyle='')
axlist[2].legend(handles=[scc3_patch, scc2_patch, scc1_patch, point1, point2],
                 loc='upper center', bbox_to_anchor=(0.88, 16), fontsize=12)
axlist[0].legend(fontsize=12, loc='upper right',bbox_to_anchor=(0.75, 1),)
axlist[0].set_xticks(pd.date_range('2004-12-29', '2005-01-31')[0::3])
axlist[0].set_xlim(pd.to_datetime('2004-12-31'), pd.to_datetime('2005-01-22'))
axlist[0].set_ylabel('[cm]', fontsize=12)
axlist[0].annotate('Hourly sea level',(0.01, 0.92), xycoords='axes fraction', ha='left', fontsize=14)
axlist[0].annotate('1.',(0.1, 0.32), xycoords='axes fraction', ha='left', fontsize=14)
axlist[0].annotate('2.',(0.21, 0.35), xycoords='axes fraction', ha='left', fontsize=14)
axlist[0].annotate('3.',(0.31, 0.42), xycoords='axes fraction', ha='left', fontsize=14)
axlist[0].annotate('4.',(0.37, 0.6), xycoords='axes fraction', ha='left', fontsize=14)
axlist[0].tick_params(axis='both', which='major', labelsize=12) # font size
axlist[1].tick_params(axis='both', which='major', labelsize=12) # font size
axlist[-1].tick_params(axis='both', which='major', labelsize=12) # font size

axlist[1].grid(True)
axlist[1].set_ylim(20,100)
axlist[1].set_ylabel('cm]', fontsize=12)
axlist[1].annotate('8-day running mean',(0.01, 0.88), xycoords='axes fraction', ha='left',va='center', fontsize=14)





plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/fig02.pdf',
            dpi=300, bbox_inches='tight')  

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/fig02.png',
            dpi=300, bbox_inches='tight')  
