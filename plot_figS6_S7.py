#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:54:06 2023

This script calculates and plots the short-term (ETC-driven) and weekly-
scale component of sea level. The weekly-scale component describles the 
volumetric changes of sea level.

The script plots Figures S6 and S7 of the upcoming GRL paper.

@author: rantanem
"""

import cluster_utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### this script reads cyclone track data and calculates various track quantities

# define tide gauge coordinates
mareographs = ['kemi','helsinki','landsort','pärnu']

abc = ['a)','b)','c)','d)']

# mareographs = ['furuögrund', 'kungholmsfort','pärnu','hamina']


# allocate dictionaries
# st = short term water level
# wb = water balance component (8-day average)
scc3_st= dict.fromkeys(mareographs)
scc1_st = dict.fromkeys(mareographs)
scc3_wb= dict.fromkeys(mareographs)
scc1_wb = dict.fromkeys(mareographs)


# input path of the clustering data
input_path = '/Users/rantanem/Documents/python/mawecli/scc_dates/'
sl_path = '/Users/rantanem/Documents/python/mawecli/Sealevel_Finland_hourlystats_N2000/'

# climatology years
years = np.arange(1980,2023)


# loop over all selected tide gauge locations
for tg in mareographs[:]:
       
    # read sea level data
    sea_level = utils.read_events(tg, sl_path, func='all').Sea_level
    # calculate water balance component as 8-day (8*24 hours) rolling mean
    wb = sea_level.rolling(window=8*24, center=True).mean()
    # Short term component is the residual
    shortTerm = sea_level - wb
    
    
    # read SCC3 onset dates
    scc3_onset_dates  = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True).index
    
    # read SCC1 onset dates
    scc1_onset_dates  = pd.read_csv(input_path +tg+'_scc1.csv', index_col=0, parse_dates=True).index
    
    
    ## CALCULATE THE SEA LEVEL ON ACTUAL SCC EVENTS
    
    scc_actual_st = pd.DataFrame(columns=scc3_onset_dates, index=np.arange(-30*24,30*24+1))
    scc_actual_wb = pd.DataFrame(columns=scc3_onset_dates, index=np.arange(-30*24,30*24+1))
    for scc_date in scc3_onset_dates:
        # Construct the lagged sea level dates +- 30 days around the actual SCC date
        scc_dates = pd.date_range(start=scc_date-pd.Timedelta('30 days'), 
                              end=scc_date+pd.Timedelta('30 days'), freq='1H')
        
        sl_scc = shortTerm.loc[scc_dates]
        wb_scc = wb.loc[scc_dates]
        
        scc_actual_st[scc_date] = sl_scc.values
        scc_actual_wb[scc_date] = wb_scc.values
    
    scc3_st[tg] = scc_actual_st
    scc3_wb[tg] = scc_actual_wb
    
    single_actual_st = pd.DataFrame(columns=scc1_onset_dates, index=np.arange(-30*24,30*24+1))
    single_actual_wb = pd.DataFrame(columns=scc1_onset_dates, index=np.arange(-30*24,30*24+1))
    for single_date in scc1_onset_dates:
        # Construct the lagged sea level dates +- 30 days around the actual SCC date
        single_dates = pd.date_range(start=single_date-pd.Timedelta('30 days'), 
                              end=single_date+pd.Timedelta('30 days'), freq='1H')
        
        sl_scc = shortTerm.loc[single_dates]
        wb_scc = wb.loc[single_dates]
        
        single_actual_st[single_date] = sl_scc.values
        single_actual_wb[single_date] = wb_scc.values
    
    scc1_st[tg] = single_actual_st
    scc1_wb[tg] = single_actual_wb
        



##### PLOT RESULTS #####    


fig, axes = plt.subplots(1,len(mareographs), figsize=(len(mareographs)*5,7), dpi=300)


for i, tg in enumerate(mareographs):
    ax = axes[i]
    

    scc1, = ax.plot(scc3_st[tg].mean(axis=1), color='red')

    scc2 = ax.fill_between(scc3_st[tg].index, 
                           scc3_st[tg].quantile(0.95, axis=1), 
                           scc3_st[tg].quantile(0.05, axis=1), 
                           facecolor='lightcoral', interpolate=True,
                           zorder=1,alpha=0.3)
    
    single1, = ax.plot(scc1_st[tg].mean(axis=1), color='blue')

    single2 = ax.fill_between(scc1_st[tg].index, 
                              scc1_st[tg].quantile(0.95, axis=1), 
                              scc1_st[tg].quantile(0.05, axis=1), 
                              facecolor='lightblue', interpolate=True,
                              zorder=1,alpha=0.3)

    

    ax.annotate(abc[i] + ' '+ tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)
    
    ax.set_ylim(-60, 60)
    
    ax.set_xticks(np.arange(-30*24, 31*24, 120))
    ax.set_xticklabels(np.char.mod('%d', np.arange(-30*24, 31*24, 120)/24))
    ax.set_xlim(-10*24, 10*24)
    
    # ax.set_xlabel('Time around the SCC onset [days]', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size
    
    ax.grid(True)
    
fig.legend([scc1, single1,], ['SCC3', 'ETC1',], 
           bbox_to_anchor=(0.6, 0.04),ncol=2, fontsize=18,
           frameon=False)
fig.text(0.5, 0.04, 'Time around the SCC onset [days]', ha='center', fontsize=16)

axes[0].set_ylabel('Sea level [cm]', fontsize=16)
    

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S06.pdf',
            dpi=300, bbox_inches='tight')      




fig, axes = plt.subplots(1,len(mareographs), figsize=(len(mareographs)*5,7), dpi=300)

for i, tg in enumerate(mareographs):
    ax = axes[i]
    

    scc1, = ax.plot(scc3_wb[tg].mean(axis=1), color='red')

    scc2 = ax.fill_between(scc3_wb[tg].index, 
                           scc3_wb[tg].quantile(0.95, axis=1), 
                           scc3_wb[tg].quantile(0.05, axis=1), 
                           facecolor='lightcoral', interpolate=True,
                           zorder=1,alpha=0.3)
    
    single1, = ax.plot(scc1_wb[tg].mean(axis=1), color='blue')

    single2 = ax.fill_between(scc1_wb[tg].index, 
                              scc1_wb[tg].quantile(0.95, axis=1), 
                              scc1_wb[tg].quantile(0.05, axis=1), 
                              facecolor='lightblue', interpolate=True,
                              zorder=1,alpha=0.3)

    
    ax.annotate(abc[i] + ' '+ tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)
    
    ax.set_ylim(-50, 85)
    ax.set_xticks(np.arange(-30*24, 31*24, 120))
    ax.set_xticklabels(np.char.mod('%d', np.arange(-30*24, 31*24, 120)/24))
    ax.set_xlim(-10*24, 10*24)
    
    # ax.set_xlabel('Time around the SCC onset [days]', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size
    
    ax.grid(True)
    
fig.legend([scc1, single1,], ['SCC3', 'ETC1',], 
           bbox_to_anchor=(0.6, 0.04),ncol=2, fontsize=18,
           frameon=False)
fig.text(0.5, 0.04, 'Time around the SCC onset [days]', ha='center', fontsize=16)

axes[0].set_ylabel('Sea level [cm]', fontsize=16)

    

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S07.pdf',
            dpi=300, bbox_inches='tight')      










      




