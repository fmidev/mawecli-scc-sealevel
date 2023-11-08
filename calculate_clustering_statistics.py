#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:38:37 2021

This script loops over the selected tide gauge locations and calculates
serial cyclone clustering statistics.

This is the main script to calculate the data for the upcoming GRL paper
"The impact of serial cyclone clustering on extremely high sea levels in the Baltic Sea"

Run this script before making the figures

@author: rantanem
"""
import cluster_utils as utils
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# define tide gauge coordinates
mareograph_coordinates = {'kemi' :        [65.67, 24.52],
                          #'oulu':         [65.04, 25.42],
                          'helsinki':     [60.15, 24.96],
                          'landsort'   :  [58.75, 17.8667],
                          'pärnu':        [58.382, 24.477],
                          #'riga':         [57.059, 24.023],
                          #'hamina':       [60.56, 27.18],
                          #'föglö':        [60.03, 20.38],
                          }


# input path of the tracks (made by Joona Corner)
input_path = '/Users/rantanem/Documents/python/mawecli/ERA5_MSLP_3hr_ONDJFM1979-2022_tracks_new/'

# Select winter seasons (1980-2022)
year1 = 1980
year2 = 2022
years = np.arange(year1,year2+1)

# radius in kilometers from which the cyclones are counted (700 km is used in literature)
radius = 700

orig_track_ids = []

# Loop over the tide gauges
for tg in list(mareograph_coordinates.keys())[:]:
    
    print(tg.capitalize()+'\n')
    
    # tige gauge coordinates
    tide_gauge_lat = mareograph_coordinates[tg][0]
    tide_gauge_lon = mareograph_coordinates[tg][1]
    
    #### read all cyclones from the track data
    all_cyclones, number_of_tracks = utils.read_all_cyclones(input_path, tide_gauge_lon, tide_gauge_lat, years, 
                                                             seasons='ONDJFM')


    ## the cyclones which pass closer than 700 km
    close_cyclones_ids = np.unique(all_cyclones[all_cyclones.distance <= radius].track_id)
    close_cyclones = all_cyclones[all_cyclones.track_id.isin(close_cyclones_ids)]
    
    # time steps when the cyclone is inside the circle
    inside_circle = close_cyclones[close_cyclones.distance<=radius]
    # select the minimum MSLP of those cyclones 
    min_mslps = inside_circle.min_mslp.groupby(inside_circle.track_id).min()
    # intensity threshold is the median MSLP of the cyclones
    slp_lim = min_mslps.median()

    # ETCs considered
    close_cyclones_ids_orig = np.unique(inside_circle[inside_circle.min_mslp<slp_lim].orig_track_id)
    orig_track_ids.append(close_cyclones_ids_orig)
   
    # calculate clustering days. Each cyclone is assigned to a day when it reaches its relative minimum pressure
    cyclone_df = utils.define_cyclone_time_series(close_cyclones, slp_lim, radius, year1, year2)
    
    print(tg.capitalize()+', threshold is '+str(slp_lim))
    print(tg.capitalize()+', in total '+str(len(min_mslps[min_mslps<slp_lim]))+' cyclones\n')
    
    # define the final SCC periods
    scc_df = utils.define_clustering_periods(cyclone_df)
    

    # label each SCC event, and if needed, remove events below the threshold lenght
    scc3_lengths, scc3_labels = utils.remove_short_scc_periods(scc_df.SCC3, threshold=0)
    scc2_lengths, scc2_labels = utils.remove_short_scc_periods(scc_df.SCC2, threshold=0)
    single_cyclone_lengths, single_cyclone_labels = utils.remove_short_scc_periods(scc_df.SCC1, threshold=0)
    
    
    # merge scc periods with the other cyclone statistics
    scc = pd.concat([cyclone_df, scc_df ], axis=1)
    
    # Find the onset dates of the SCC events
    scc3_onset_dates = scc3_labels[~scc3_labels.duplicated(keep='first')][1:]
    scc3_onset_dates = scc3_onset_dates.rename('SCC3')

    scc2_onset_dates = scc2_labels[~scc2_labels.duplicated(keep='first')][1:]
    scc2_onset_dates = scc2_onset_dates.rename('SCC2')
    
    single_onset_dates = single_cyclone_labels[~single_cyclone_labels.duplicated(keep='first')][1:]
    single_onset_dates = single_onset_dates.rename('SCC1')
    
    # merge SCC onset date & length
    scc3_lengths = pd.Series(index=scc3_onset_dates.index, data=scc3_lengths)
    scc2_lengths = pd.Series(index=scc2_onset_dates.index, data=scc2_lengths)
    single_lengths = pd.Series(index=single_onset_dates.index, data=single_cyclone_lengths)
    
    # Find the ids of those cyclones which participate the clustering
    scc3_ids = utils.define_clustering_ids(cyclone_df.track_ids, scc3_lengths, scc3_onset_dates)
    
    # Date of SCC periods, duration, and intensity of ETCs within the period
    scc1 = utils.calculate_cluster_intensities(cyclone_df.mslp, single_lengths, single_onset_dates)
    scc2 = utils.calculate_cluster_intensities(cyclone_df.mslp, scc2_lengths, scc2_onset_dates)
    scc3 = utils.calculate_cluster_intensities(cyclone_df.mslp, scc3_lengths, scc3_onset_dates)
    
    # output the scc onset dates
    scc3.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc3.csv')
    scc2.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc2.csv')
    scc1.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc1.csv')


    # output the clustering statistics
    scc.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc_statistics.csv')
    
    # output the close cyclones
    close_cyclones.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_close_cyclones.csv')
    
    scc1_850vo = utils.calculate_cluster_intensities(cyclone_df['850vo'], single_lengths, single_onset_dates)
    scc2_850vo = utils.calculate_cluster_intensities(cyclone_df['850vo'], scc2_lengths, scc2_onset_dates)
    scc3_850vo = utils.calculate_cluster_intensities(cyclone_df['850vo'], scc3_lengths, scc3_onset_dates)
    # output the scc onset dates
    scc1_850vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc1_850vo.csv')
    scc2_850vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc2_850vo.csv')
    scc3_850vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc3_850vo.csv')
    
    scc1_t63vo = utils.calculate_cluster_intensities(cyclone_df['t63vo'], single_lengths, single_onset_dates)
    scc2_t63vo = utils.calculate_cluster_intensities(cyclone_df['t63vo'], scc2_lengths, scc2_onset_dates)
    scc3_t63vo = utils.calculate_cluster_intensities(cyclone_df['t63vo'], scc3_lengths, scc3_onset_dates)
    # output the scc onset dates
    scc1_t63vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc1_t63vo.csv')
    scc2_t63vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc2_t63vo.csv')
    scc3_t63vo.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc3_t63vo.csv')
    
    scc1_925ws6 = utils.calculate_cluster_intensities(cyclone_df['925ws6'], single_lengths, single_onset_dates)
    scc2_925ws6 = utils.calculate_cluster_intensities(cyclone_df['925ws6'], scc2_lengths, scc2_onset_dates)
    scc3_925ws6 = utils.calculate_cluster_intensities(cyclone_df['925ws6'], scc3_lengths, scc3_onset_dates)
    # output the scc onset dates
    scc1_925ws6.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc1_925ws6.csv')
    scc2_925ws6.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc2_925ws6.csv')
    scc3_925ws6.to_csv('/Users/rantanem/Documents/python/mawecli/scc_dates/'+tg+'_scc3_925ws6.csv')


