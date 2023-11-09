#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:25:54 2023

This script plots figures S01 and S02 

@author: rantanem
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = np.round(rect.get_height(),1)

        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)


### this script reads cyclone track data and calculates various track quantities

# define tide gauge coordinates
mareographs = ['kemi', 'helsinki', 'landsort', 'pärnu']
abc=['a)', 'b)','c)','d)']


# allocate dictionary
mareograph_scc3 = dict.fromkeys(mareographs)
mareograph_scc2 = dict.fromkeys(mareographs)
mareograph_scc3_lengths = dict.fromkeys(mareographs)
mareograph_scc2_lengths = dict.fromkeys(mareographs)
mareograph_scc3_cyclones = dict.fromkeys(mareographs)
mareograph_scc2_cyclones = dict.fromkeys(mareographs)




# input path of the clustering data
input_path = '/Users/rantanem/Documents/python/mawecli/scc_dates/'





for tg in mareographs[:]:

    # read SCC onset dates
    scc3_onset_dates  = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True)
    scc2_onset_dates  = pd.read_csv(input_path +tg+'_scc2.csv', index_col=0, parse_dates=True)
        
    mareograph_scc3[tg] = scc3_onset_dates.index
    
    mareograph_scc3_lengths[tg] = scc3_onset_dates.duration.values.squeeze()    
    
    # read SCC onset dates
    
        
    mareograph_scc2[tg] = scc2_onset_dates.index
    
    mareograph_scc2_lengths[tg] = scc2_onset_dates.duration.values.squeeze()
    
    # read SCC3 cyclones 
    scc3_cyclones  = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True).iloc[:, 1:].count(axis=1)
    mareograph_scc3_cyclones[tg] = scc3_cyclones.values.squeeze()
    
    # read SCC cyclones 
    scc2_cyclones  = pd.read_csv(input_path +tg+'_scc2.csv', index_col=0, parse_dates=True).iloc[:, 1:].count(axis=1)
    mareograph_scc2_cyclones[tg] = scc2_cyclones.values.squeeze()

    

    
    




fig, axes = plt.subplots(1,len(mareographs), figsize=(len(mareographs)*5,5), dpi=300)

for i, tg in enumerate(mareographs):
    ax = axes[i]
    
    x = mareograph_scc3_lengths[tg]
    y = mareograph_scc3_cyclones[tg]
    
    ax.scatter(x,y, s=80, zorder=4)
    
    r = np.corrcoef(x,y)[0][1]
    print(tg, r)
    
    
    ax.grid(True)
    
    ax.axvline(x=np.median(mareograph_scc3_lengths[tg]), color='k')
    meanstr = np.round(np.median(mareograph_scc3_lengths[tg]),1)
    ax.annotate(str(meanstr), (meanstr-2.,11),xycoords='data',rotation=90, fontsize=15)
    
    ax.axhline(y=np.median(mareograph_scc3_cyclones[tg]), color='k')
    meanstr = np.round(np.median(mareograph_scc3_cyclones[tg]),1)
    ax.annotate(str(meanstr), (24,meanstr+0.1,),xycoords='data',rotation=0, fontsize=15)

    
    ax.set_xticks(np.arange(0,38,5))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=17) # font size
    
    ax.set_xlim(0,30)
    ax.set_ylim(0,13)
    
    ax.set_xlabel('Length of SCC3 period [days]', fontsize=16)
    
    
    ax.annotate(abc[i] + ' ' +tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', 
                fontsize=18)

axes[0].set_ylabel('Number of cyclones', fontsize=16)
plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S01.pdf',dpi=300, bbox_inches='tight')

 

fig, axes = plt.subplots(1,len(mareographs), figsize=(len(mareographs)*5,5), dpi=300)

for i, tg in enumerate(mareographs):
    ax = axes[i]
    

    
    x = mareograph_scc2_lengths[tg]
    y = mareograph_scc2_cyclones[tg]
    
    ax.scatter(x,y, s=80, zorder=4)
    
    r = np.corrcoef(x,y)[0][1]
    print(tg, r)
    
    
    ax.grid(True)
    
    ax.axvline(x=np.median(mareograph_scc2_lengths[tg]), color='k')
    meanstr = np.round(np.median(mareograph_scc2_lengths[tg]),1)
    ax.annotate(str(meanstr), (meanstr-2.,5),xycoords='data',rotation=90, fontsize=15)
    print(np.percentile(mareograph_scc2_lengths[tg], 95))
    
    ax.axhline(y=np.median(mareograph_scc2_cyclones[tg]), color='k')
    meanstr = np.round(np.median(mareograph_scc2_cyclones[tg]),1)
    ax.annotate(str(meanstr), (21.5,meanstr+0.1,),xycoords='data',rotation=0, fontsize=15)

    
    ax.set_xticks(np.arange(0,38,5))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=17) # font size
    
    ax.set_xlim(0,27)
    ax.set_ylim(0,6)
    
    ax.set_xlabel('Length of SCC2 period [days]', fontsize=16)
    
    
    ax.annotate(abc[i] + ' ' +tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', 
                fontsize=18)

axes[0].set_ylabel('Number of cyclones', fontsize=16)
plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S02.pdf',dpi=300, bbox_inches='tight')