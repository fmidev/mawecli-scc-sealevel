#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:16:06 2023

Python script to plot Figure 3 in GRL paper "The impact of serial cyclone clustering 
on extremely high sea levels in the Baltic Sea"



@author: rantanem
"""

import cluster_utils as utils
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_peak_storm_surge(sl_onset_dates, scc_onset_dates):
    
    scc_max_df = pd.DataFrame(index=scc_onset_dates.index, columns=['Maximum', 'DateMax'])
    sl_events_after_scc = []

    # loop over SCC3 events
    for scc_date in scc_onset_dates.index:
        
        # length of SCC3 period
        l = int(scc_onset_dates.loc[scc_date][0])
        
        # Select dates during the SCC3 periods
        maxdates = pd.date_range(start=scc_date, end=scc_date+pd.Timedelta(str(l-1)+' days'))
        
        # check if sea level event has occurred within that time frame
        a = any(np.isin(sl_onset_dates, maxdates))
        if a:
            sl_events_after_scc.append(sl_onset_dates[np.isin(sl_onset_dates, maxdates)])
        
        # read the peak storm surge during those dates
        sl_max = sea_level.loc[maxdates].max()
        
        # read the date of peak storm surge
        sl_maxdate = sea_level.loc[maxdates].idxmax()
        
        
        scc_max_df.loc[scc_date]['Maximum'] = sl_max
        scc_max_df.loc[scc_date]['DateMax'] = sl_maxdate
    
    sl_events_after_scc = pd.DatetimeIndex(np.unique(np.hstack(sl_events_after_scc)))
        
    return sl_events_after_scc, scc_max_df

def get_lagged_sea_level(scc_onset_dates):
    
    scc_actual_df = pd.DataFrame(columns=scc_onset_dates, index=np.arange(-30,31))
    
    for scc_date in scc_onset_dates:
        # Construct the lagged sea level dates +- 30 days around the actual SCC date
        scc_dates = pd.date_range(start=scc_date-pd.Timedelta('30 days'), 
                              end=scc_date+pd.Timedelta('30 days'))
        
        sl_scc = sea_level.loc[scc_dates].rolling(window=1, center=True).mean()
        
        scc_actual_df[scc_date] = sl_scc.values
        
    return scc_actual_df

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = np.round(rect.get_height(),1)

        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)
        
        

# define tide gauge coordinates
mareographs = ['kemi', 'helsinki','landsort', 'pärnu']
# mareographs = ['oulu','hamina','Föglö','riga']




abc = ['a)','b)','c)','d)']

efg = ['e)','f)','g)','h)']

ijk = ['i)','j)','k)','l)']


# allocate dictionaries
peak_surges_scc3 = dict.fromkeys(mareographs)
peak_surges_scc2 = dict.fromkeys(mareographs)
peak_surges_scc1 = dict.fromkeys(mareographs)


# allocate dictionaries
mareograph_scc3_climatology = dict.fromkeys(mareographs)
mareograph_scc2_climatology = dict.fromkeys(mareographs)
mareograph_scc1_climatology = dict.fromkeys(mareographs)
mareograph_scc3_sealevel = dict.fromkeys(mareographs)
mareograph_scc2_sealevel = dict.fromkeys(mareographs)
mareograph_scc1_sealevel = dict.fromkeys(mareographs)

thresholds = {}


# input path of the clustering data
input_path = '/Users/rantanem/Documents/python/mawecli/scc_dates/'
sealevel_path = '/Users/rantanem/Documents/python/mawecli/Sealevel_Finland_hourlystats_N2000/'

# number of bootstrapping in climatological sea level
N = 10

# climatology years
years = np.arange(1980,2023)



# loop over all selected tide gauge locations
for tg in mareographs[:]:
    
    # read sea level data
    sea_level = utils.read_events(tg, sealevel_path, func='max').Sea_level
    
    
    # read SCC3 onset dates
    scc3 = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True)
    scc3_onset_dates  = scc3.index
    
    # read SCC2 onset dates
    scc2_onset_dates  = pd.read_csv(input_path +tg+'_scc2.csv', index_col=0, parse_dates=True).index
    
    # read SCC1/ETC1 onset dates
    scc1_onset_dates  = pd.read_csv(input_path +tg+'_scc1.csv', index_col=0, parse_dates=True).index
    
    # get the climatologies       
    mareograph_scc3_climatology[tg] = utils.calculate_sealevel_climatology(N, scc3_onset_dates, sea_level, lag=30)
    mareograph_scc2_climatology[tg] = utils.calculate_sealevel_climatology(N, scc2_onset_dates, sea_level, lag=30)
    mareograph_scc1_climatology[tg] =  utils.calculate_sealevel_climatology(N, scc1_onset_dates, sea_level, lag=30)
    
    
    ## CALCULATE THE SEA LEVEL ON ACTUAL SCC EVENTS
    mareograph_scc3_sealevel[tg] = get_lagged_sea_level(scc3_onset_dates)
    mareograph_scc2_sealevel[tg] = get_lagged_sea_level(scc2_onset_dates)
    mareograph_scc1_sealevel[tg] = get_lagged_sea_level(scc1_onset_dates)
    


# percentages
percetages_df = pd.DataFrame(index=mareographs, columns=['SCC3','SCC2', 'ETC1', 'No\nstrong ETCs'])

# loop over all selected tide gauge locations
for tg in mareographs[:]:
    
    print('\n'+tg.capitalize()+':\n')
    
    
    # read sea level data
    sea_level_data = utils.read_events(tg, sealevel_path, func='max')
    sea_level = sea_level_data.Sea_level.loc[slice('1979-09-01', '2022-05-31')]
    
    # define extreme sea level event
    sl_events = sea_level[(sea_level.index.month<=3)|(sea_level.index.month>=10)]
    sl_event = np.nanpercentile(sl_events, 98)
    print('\nExtreme sea level: '+str(np.round(sl_event,1))+' cm')
    thresholds[tg] = sl_event
    
    # read extreme sea level events
    sl_onset_dates  = pd.read_csv(input_path +tg+'_sl_onset_dates.csv', index_col=0, parse_dates=True).index
    
    
    # read SCC3 onset dates
    scc3_onset_dates  = pd.read_csv(input_path +tg+'_scc3.csv', index_col=0, parse_dates=True)
    
    # read SCC2 onset dates
    scc2_onset_dates  = pd.read_csv(input_path +tg+'_scc2.csv', index_col=0, parse_dates=True)
    
    # read single cyclone onset dates
    scc1_onset_dates  = pd.read_csv(input_path +tg+'_scc1.csv', index_col=0, parse_dates=True)
    
    # read scc stats 
    scc  = pd.read_csv(input_path +tg+'_scc_statistics.csv', index_col=0, parse_dates=True)
    scc = pd.concat([scc, sea_level], axis=1)
    

    
    sl_events_after_scc3,scc3_max_df =  get_peak_storm_surge(sl_onset_dates, scc3_onset_dates)
    sl_events_after_scc2,scc2_max_df =  get_peak_storm_surge(sl_onset_dates, scc2_onset_dates)
    sl_events_after_scc1,scc1_max_df =  get_peak_storm_surge(sl_onset_dates, scc1_onset_dates)


    # Save peak storm surges for plotting
    peak_surges_scc3[tg] = scc3_max_df
    peak_surges_scc2[tg] = scc2_max_df
    peak_surges_scc1[tg] = scc1_max_df
    
    


    # Calculate extreme sea level events which are not in SCC3, SCC2 or SCC1
    sl_events_left = sl_onset_dates[~np.isin(sl_onset_dates, sl_events_after_scc3.union(sl_events_after_scc2).union(sl_events_after_scc1))]

    # percentage of sea level events occurring after SCC and SC onset dates
    scc3_per = (len(sl_events_after_scc3) / len(sl_onset_dates)) * 100
    scc2_per = (len(sl_events_after_scc2) / len(sl_onset_dates)) * 100
    scc1_per = (len(sl_events_after_scc1) / len(sl_onset_dates)) * 100
    left_per = (len(sl_events_left) / len(sl_onset_dates)) * 100
    
    no_cyclones_df = pd.Series(index=sl_events_left,data=np.ones(len(sl_events_left)))
    
    # save SL events which have been happened without ETCs
    no_cyclones_df.to_csv(input_path + tg+'_no_cyclones_sealevel_events.csv')

    
    print('\nPercentage of extreme sea level events after single cyclones: '+str(np.round(scc1_per,1))+' %')
    print('\nPercentage extreme sea level events after SCC: '+str(np.round(scc3_per,1))+' %\n')
    print('\nPercentage extreme sea level events in nether group: '+str(np.round(left_per,1))+' %\n')
    percetages_df.loc[tg] = scc3_per, scc2_per, scc1_per, left_per


    # probability of extreme sea level events:
    p_single = np.sum(scc1_max_df['Maximum'] > sl_event) / len(scc1_max_df['Maximum'])*100
    p_scc = np.sum(scc3_max_df['Maximum'] > sl_event) / len(scc3_max_df['Maximum'])*100
    # print('\nProbability of extreme sea level event after single cyclones: '+str(np.round(p_single,1))+' %')
    # print('\nProbability of extreme sea level event after SCC: '+str(np.round(p_scc,1))+' %\n')
    print('#############')


### PLOT THE RESULTS #####


fig, axes = plt.subplots(3,len(mareographs), figsize=(len(mareographs)*5,15), dpi=300)
axes = np.ravel(axes)[:]
plt.subplots_adjust(wspace=0.3, hspace=0.35, top=0.95, bottom=0.02)


for i, tg in enumerate(mareographs):
    ax = axes[i]
    

    diff = mareograph_scc3_sealevel[tg].mean(axis=1) - mareograph_scc3_climatology[tg].mean(axis=1)

    diff_95 = mareograph_scc3_sealevel[tg].quantile(0.95,axis=1) - mareograph_scc3_climatology[tg].mean(axis=1)
    diff_05 = mareograph_scc3_sealevel[tg].quantile(0.05,axis=1) - mareograph_scc3_climatology[tg].mean(axis=1)

    scc1, = ax.plot(diff, color='green', linewidth=2.5)
    
    ax.plot(diff_95, color='green', linewidth=1, linestyle='--')
    ax.plot(diff_05, color='green', linewidth=1, linestyle='--')

    scc2 = ax.fill_between(mareograph_scc3_sealevel[tg].index, diff_95, diff_05, 
                           facecolor='green', interpolate=True,
                           zorder=1,alpha=0.2)
    
    
    
    diff = mareograph_scc2_sealevel[tg].mean(axis=1) - mareograph_scc2_climatology[tg].mean(axis=1)
    
    diff_95 = mareograph_scc2_sealevel[tg].quantile(0.95,axis=1) - mareograph_scc2_climatology[tg].mean(axis=1)
    diff_05 = mareograph_scc2_sealevel[tg].quantile(0.05,axis=1) - mareograph_scc2_climatology[tg].mean(axis=1)

    two1, = ax.plot(diff, color='orange', linewidth=2.5)
    ax.plot(diff_95, color='orange', linewidth=1, linestyle='--')
    ax.plot(diff_05, color='orange', linewidth=1, linestyle='--')
    
    two2 = ax.fill_between(mareograph_scc3_sealevel[tg].index, diff_95, diff_05, 
                           facecolor='orange', interpolate=True,
                           zorder=1,alpha=0.2)
    
    diff = mareograph_scc1_sealevel[tg].mean(axis=1) - mareograph_scc1_climatology[tg].mean(axis=1)
    
    
    diff_95 = mareograph_scc1_sealevel[tg].quantile(0.95,axis=1) - mareograph_scc1_climatology[tg].mean(axis=1)
    diff_05 = mareograph_scc1_sealevel[tg].quantile(0.05,axis=1) - mareograph_scc1_climatology[tg].mean(axis=1)

    single1, = ax.plot(diff, color='blue', linewidth=2.5)
    ax.plot(diff_95, color='blue', linewidth=1, linestyle='--')
    ax.plot(diff_05, color='blue', linewidth=1, linestyle='--')
    
    single2 = ax.fill_between(mareograph_scc3_sealevel[tg].index, diff_95, diff_05, 
                              facecolor='blue', interpolate=True,
                              zorder=1,alpha=0.2)
    
    ax.axhline(y=0, color='k')

    
    ax.annotate(abc[i] + ' '+ tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)
    
    ax.set_ylim(-75, 110)

    ax.set_xticks(np.arange(-30, 40, 10))
    ax.set_xlim(-20, 20)
    
    
    # ax.set_xlabel('Time around the SCC onset [days]', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size
    
    ax.grid(True)
    
fig.legend([scc1, two1, single1],
           ['SCC3', 'SCC2', 'ETC1'], 
           bbox_to_anchor=(0.65, 1),ncol=3, fontsize=18,
           frameon=False)

fig.text(0.5, 0.66, 'Time around the SCC onset [days]', fontsize=18,ha='center')

axes[0].set_ylabel('Daily maximum sea level anomaly [cm]', fontsize=18)




for i, tg in enumerate(mareographs):
    ax = axes[i+4]
    
    bins = np.arange(-100, 290, 10)
    width=10
    
    
    single = ax.hist(peak_surges_scc1[tg]['Maximum'].astype(float), bins=bins,facecolor='turquoise', width=width, edgecolor='k',
            density=True, alpha=0.6, zorder=4)
    
    single_mean = peak_surges_scc1[tg]['Maximum'].mean()
    ax.axvline(x=single_mean, linestyle='--', color='turquoise')
    ax.annotate(str(int(np.round(single_mean,0))), (single_mean, 0.023), xycoords='data', 
                fontsize=18, color='turquoise', fontweight='bold', rotation=90, ha='right')
    
    SCC = ax.hist(peak_surges_scc3[tg]['Maximum'].astype(float), bins=bins, facecolor='coral',width=width, edgecolor='k',
            density=True, alpha=0.6, zorder=4)
    
    scc_mean = peak_surges_scc3[tg]['Maximum'].mean()
    ax.axvline(x=scc_mean, linestyle='--', color='coral')
    ax.annotate(str(int(np.round(scc_mean,0))), (scc_mean, 0.027), xycoords='data', 
                fontsize=18, color='coral', fontweight='bold', rotation=90, ha='right')

    
    ax.set_xticks(bins[::5])
    
    ax.set_ylim(0, 0.03)
    ax.set_xlim(-50,285)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size
    
    ax.annotate(efg[i]+' '+tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)

    ax.axvline(x=thresholds[tg], linestyle='--', color='k', zorder=3)
    
    idx1 = np.isfinite(peak_surges_scc1[tg]['Maximum'].astype(float))
    idx2 = np.isfinite(peak_surges_scc3[tg]['Maximum'].astype(float))
    
    res = stats.kstest(peak_surges_scc1[tg]['Maximum'].astype(float).values, peak_surges_scc3[tg]['Maximum'].astype(float).values)

axes[4].set_ylabel('Density', fontsize=16)
fig.text(0.5, 0.32, 'Peak storm surge [cm]', fontsize=18,ha='center')


#create legend
handles = [Rectangle((0,0),1,1,color=c,ec="k", alpha=0.6) for c in ['coral','turquoise']]
labels= ['SCC3',"ETC1",]
fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.61),ncol=1, fontsize=18,
           frameon=False)



for i, tg in enumerate(mareographs):
    ax = axes[i+8]
    
    x = [0,1,2,3]
    rects1 = ax.bar(x=x, height=percetages_df.loc[tg], color=['green', 'orange', 'blue', 'grey'], zorder=5)
    
    ax.set_xticks(ticks=x, labels=percetages_df.columns)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size

    ax.annotate(ijk[i] + ' ' + tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)
    
    ax.grid()
    
    autolabel(rects1)
    
    ax.set_ylim(0,70)
axes[8].set_ylabel('Percentage of extreme\nsea level events [%]', fontsize=16)

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure04.pdf',
            dpi=300, bbox_inches='tight')  

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure04.png',
            dpi=300, bbox_inches='tight')  







fig, axes = plt.subplots(2,2, figsize=(len(mareographs)*2.5,12), dpi=300, sharey=True)
axes = np.ravel(axes)

for i, tg in enumerate(mareographs):
    ax = axes[i]
    
    x = [0,1,2,3]
    rects1 = ax.bar(x=x, height=percetages_df.loc[tg], color=['green', 'orange', 'blue', 'grey'], zorder=5)
    
    ax.set_xticks(ticks=x, labels=percetages_df.columns)
    
    ax.tick_params(axis='both', which='major', labelsize=16) # font size

    ax.annotate(abc[i] + ' ' + tg.capitalize(), (0.01, 1.02), xycoords='axes fraction', fontsize=16)
    
    ax.grid()
    
    autolabel(rects1)
    
    ax.set_ylim(0,70)

fig.text(0.04, 0.5, 'Percentage of extreme sea level events [%]', fontsize=18, va='center', rotation='vertical')


plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure04_poster.pdf',
            dpi=300, bbox_inches='tight')  

plt.savefig('/Users/rantanem/Documents/python/mawecli/figures/figure04_poster.png',
            dpi=300, bbox_inches='tight')  

