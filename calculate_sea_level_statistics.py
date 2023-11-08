#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:38:37 2021

This script calculates the extreme sea level events used in the upcoming GRL paper
"The impact of serial cyclone clustering on extremely high sea levels in the Baltic Sea"


@author: rantanem
"""
import cluster_utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


### this script reads cyclone track data and calculates various sea level quantities

# define tide gauge coordinates

mareographs = ['kemi', 'helsinki', 'landsort', 'pärnu']
# mareographs = ['oulu','hamina','riga','föglö']

abc=['a)', 'b)','c)','d)']

mareograph_sl = dict.fromkeys(mareographs)

# input and output paths of the sealevel
input_path = '/Users/rantanem/Documents/python/mawecli/Sealevel_Finland_hourlystats_N2000/'
output_path = '/Users/rantanem/Documents/python/mawecli/daily_sealevel/'


years = np.arange(1980,2023)

func='max'

for tg in mareographs[:]:
       
    # read sea level data
    df = utils.read_events(tg, input_path, func=func).Sea_level
    df = df.loc[slice('1979-10-01', '2022-03-31')]
    
    # save daily sea level
    df.to_csv(output_path + tg+'_daily_'+func+'_sealevel.csv')

    # define extreme sea level event
    threshold = np.nanpercentile(df[(df.index.month<=3)|(df.index.month>=10)], 98)

    print('Extreme sea level: '+str(np.round(threshold,1))+' cm')
    
    sl_events = utils.define_extreme_sl_events(df, threshold)

    
    # label each sea level event, and if needed, remove events below the threshold lenght
    sl_lengths, sl_labels = utils.remove_short_scc_periods(sl_events, threshold=0)

    
    # Find the onset dates of the sea level events
    sl_onset_dates = sl_labels[~sl_labels.duplicated(keep='first')][1:]
    sl_onset_dates = sl_onset_dates.rename('SL period')
    
    print('Number of extreme sea level events: ' +str(len(sl_onset_dates)) + '\n')
    
    # Find the maximum sea level per each sea level event
    max_sea_level = utils.define_maximum_sealevel(df, sl_onset_dates, sl_lengths)
    
    sl_lengths = pd.Series(index=sl_onset_dates.index, data=sl_lengths)
    
    sl = pd.concat([max_sea_level, sl_lengths.rename('length')], axis=1)
    mareograph_sl[tg] = sl.length
    
    # output the sea level events
    sl.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_sl_onset_dates.csv')





fig, axes = plt.subplots(1,len(mareographs), figsize=(len(mareographs)*5,5), dpi=300)

for i, tg in enumerate(mareographs):
    ax = axes[i]
    
    hist, bin_edges = np.histogram(mareograph_sl[tg].values,bins=np.arange(0.5,30.5, 1))
    
    x = (bin_edges[:-1]+0.5)
    y = hist
    
    rects1 = ax.bar(x, y, width=0.7)
        
    ax.set_ylim(0,41)
    ax.set_xlim(0, 29)
    
    ax.annotate(str(np.sum(hist)) + ' events', (0.54,0.94),xycoords='axes fraction',fontsize=14 )
    ax.annotate('Median: '+str(int(np.median(mareograph_sl[tg].values)))+' days', (0.54,0.87),xycoords='axes fraction', 
                fontsize=14)
    
    ax.annotate('95%: '+str(int(np.percentile(mareograph_sl[tg].values, 95)))+' days', (0.54,0.8),xycoords='axes fraction', 
                fontsize=14)
    
    ax.set_xticks(np.arange(0,30,3))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=16) # font size
    
    ax.set_xlabel('Duration of extreme\nsea level events [days]', fontsize=16)
    
    ax.annotate(abc[i] + ' ' +tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', 
                fontsize=18)

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S03.png',dpi=300, bbox_inches='tight')
plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure_S03.pdf',dpi=300, bbox_inches='tight')






tg='helsinki'
# read sea level data
df = utils.read_events(tg, input_path, func='max').Sea_level
df = df.loc[slice('1979-10-01', '2022-03-31')]

# define extreme sea level event
threshold = np.nanpercentile(df[(df.index.month<=3)|(df.index.month>=10)], 98)

sl_events = utils.define_extreme_sl_events(df, threshold)
# label each sea level event, and if needed, remove events below the threshold lenght
sl_lenghts, sl_labels = utils.remove_short_scc_periods(sl_events*1.0, threshold=0)


# Find the onset dates of the sea level events
sl_onset_dates = sl_labels[~sl_labels.duplicated(keep='first')][1:]
sl_onset_dates = sl_onset_dates.rename('SL period')
  


d1 = '2014-12-29'
d2 = '2015-01-31'
sample = df.loc[slice(d1,d2)]
sl_sample = sl_events.loc[slice(d1,d2)]

fig, ax = plt.subplots(1,1, figsize=(9,6), dpi=300, sharey=True)

ax.plot(sample, '-o')
ax.scatter(sample[sample>threshold].index,sample[sample>threshold].values,s=50, color='red', zorder=5,
           label='Extremely high sea level days')
ax.axhline(y=threshold, color='red', linestyle='--')

ax2 = ax.twinx()
ax2.bar(sl_sample.index, sl_sample,facecolor='orange')

ax2.bar(sl_onset_dates.index, np.ones(len(sl_onset_dates.index,)), facecolor='red', zorder=6,
        label='Extreme sea level\nevent onset dates')

ax.set_ylabel('Daily maximum sea level [cm]', fontsize=16)


ax.tick_params(axis='y', which='major', labelsize=16) # font size
ax.tick_params(axis='x', which='major', labelsize=14) # font size
ax2.tick_params(axis='y', which='major', labelsize=16) # font size

ax.set_ylim(0,120)
ax2.set_ylim(0,6)
ax2.set_yticks((0,1),['No event','Event'])
ax.set_xlim(pd.to_datetime(d1),pd.to_datetime(d2))

myFmt = mdates.DateFormatter('%d.%m.\n%Y')
ax.xaxis.set_major_formatter(myFmt)

ax.grid(True)
ax.legend(loc='upper left')
ax2.legend()

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/sea_level_example.png',
            dpi=300, bbox_inches='tight')    