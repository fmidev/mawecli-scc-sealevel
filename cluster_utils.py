#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:25:40 2023

All the subroutines / functions related to cyclone clustering calculations

@author: rantanem
"""
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians

def get_distance(lat1, lon1, lat2, lon2):
    
    R = 6370
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
            
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance


def read_all_cyclones(input_path, tide_gauge_lon, tide_gauge_lat, years, seasons):
    
    # This function reads cyclone track data and returns the tracks as a dataframe
    
    all_cyclones_list = []
    
    new_track_id = 1
    number_of_tracks = []
    
    ## loop over the seasons (years)
    for y in years:
    
        # read all the tracks from the specific season
        if seasons=='ONDJFM': 
            chosen_year = str(y-1)+str(y)
            chosen_season = 'ONDJFM'

            datadir = input_path + 'ERA5_MSLP_3hr_'+chosen_season+chosen_year+'/'

            track_file = datadir + 'ff_trs_neg_mslp4.0deg_wg3.0deg_wg6.0deg'
            track_file = datadir + \
            'ff_trs_neg_mslp4.0deg_wg3.0deg_wg6.0deg_850vo4.0deg_T63vo4.0deg_avg925ws3.0deg_avg925ws6.0deg'
    
            df_tracks1 = pd.read_csv(track_file,skiprows=2,header=None) # skip first two rows which contain zeros
            tr_num_tot1 =  int(df_tracks1.iloc[0].str.split('\s+').tolist()[0][1])
            df_tracks1 = df_tracks1.iloc[1:]
            
            df_tracks = pd.concat([df_tracks1])
            tr_num_tot = tr_num_tot1
            
            print('Cyclones in '  + chosen_season +' ' + chosen_year[:4] + '-' + chosen_year[4:] )

        elif seasons=='AMJJAS':
            chosen_year = str(y)
            chosen_season = 'AMJJAS'

            datadir = input_path + 'ERA5_MSLP_3hr_'+chosen_season+chosen_year+'/'

            track_file = datadir + 'ff_trs_neg_mslp4.0deg_wg3.0deg_wg6.0deg'
            track_file = datadir + \
            'ff_trs_neg_mslp4.0deg_wg3.0deg_wg6.0deg_850vo4.0deg_T63vo4.0deg_avg925ws3.0deg_avg925ws6.0deg'
    
    
            df_tracks2 = pd.read_csv(track_file,skiprows=2,header=None) # skip first two rows which contain zeros
            tr_num_tot2 =  int(df_tracks2.iloc[0].str.split('\s+').tolist()[0][1])
            df_tracks2 = df_tracks2.iloc[1:]
            
            df_tracks = pd.concat([df_tracks2])
            tr_num_tot = tr_num_tot2
            
            print('Cyclones in '  + chosen_season +' ' + chosen_year[:4] )

        
        
        # print('In total ' + str(tr_num_tot) + ' cyclones\n')
        number_of_tracks.append(tr_num_tot)
    

        # Loop through all tracks of the specific season
        df_tracks_loop = df_tracks.copy() # copy of all tracks, remove track after each loop
        for tr in range(0,tr_num_tot):
        
            # Track ID and number of time steps: save and remove rows
            point_num = int(df_tracks_loop.iloc[1].str.split('\s+').tolist()[0][1])
            orig_track_id = df_tracks_loop.iloc[0].str.split('\s+').tolist()[0][1] + \
                            '_' + df_tracks_loop.iloc[0].str.split('\s+').tolist()[0][3]
        
            df_tracks_loop = df_tracks_loop.iloc[2:]
        
            # New dataframe for this track (NOTE: two different separators so the splitting is done twice)
            df_tr = pd.DataFrame(df_tracks_loop[:point_num][0].str.split('&').tolist())
            df_split = pd.DataFrame(df_tr[0].str.split('\s+',expand=False).tolist(),columns=['a','b','c','d','e'])
            del df_split['e'] # delete e column (empty space)
            df_tr = pd.concat([df_split,df_tr],axis=1)
            del df_tr[0],df_tr[18] # delete 0 (contains splitted columns) and 10 (empty space) columns
            df_tr.columns = ['time','lon_centre','lat_centre','mslp_anom','lon_mslp','lat_mslp','min_mslp',
                             'lon_wg_3','lat_wg_3','wg_3','lon_wg_6','lat_wg_6','wg_6', 
                             'lon1', 'lat1', '850vo', 'lon2', 'lat2', 't63vo','925ws3','925ws6']
            df_tr['datetime'] = pd.to_datetime(df_tr['time'],format='%Y%m%d%H')  
        
        
            # convert longitude from 0..360 to -180...180
            df_tr['lon_mslp'][df_tr['lon_mslp'].astype(float) > 360] = np.nan
            new_lon = ((df_tr.lon_mslp.astype(float) + 180) % 360) - 180
            df_tr['lon_mslp'] = new_lon
        
        
            new_lon = ((df_tr.lon_centre.astype(float) + 180) % 360) - 180
            df_tr['lon_centre'] = new_lon
           
            # Change type to float
            df_tr['lat_centre'] = df_tr['lat_centre'].astype(float)
            df_tr['min_mslp'] = df_tr['min_mslp'].astype(float)
            df_tr['wg_6'] = df_tr['wg_6'].astype(float)
            df_tr['850vo'] = df_tr['850vo'].astype(float)
            df_tr['t63vo'] = df_tr['t63vo'].astype(float)
            df_tr['925ws3'] = df_tr['925ws3'].astype(float)
            df_tr['925ws6'] = df_tr['925ws6'].astype(float)
            df_tr['lat_mslp'] = df_tr['lat_mslp'].astype(float)
            df_tr['lat_mslp'][df_tr['lat_mslp'] > 90] = np.nan
            
        
            # add column depicting the distance to the tide gauge
            cyclone_lat = df_tr.lat_centre
            cyclone_lon = df_tr.lon_centre
        
            dist = []
            # Calculate the distance to the tide gauge at all time steps of the track
            for i in np.arange(0, len(df_tr)):
            
                lat1 = radians(tide_gauge_lat)
                lon1 = radians(tide_gauge_lon)
                lat2 = radians(cyclone_lat[i])
                lon2 = radians(cyclone_lon[i])
            
                distance = np.round(get_distance(lat1, lon1, lat2, lon2), 0)
                dist.append(distance)
            
            df_tr['distance'] = np.ravel(dist)
            

            # add unique track ID
            df_tr['track_id'] = new_track_id
            df_tr['orig_track_id'] = orig_track_id
            
            # Change the format of time
            tr_times = pd.to_datetime(df_tr.time.values, format='%Y%m%d%H')
            df_tr.index=tr_times
        
        
            ## append track dataframe which only has the most relevant data
            df_tr_new = df_tr[['lon_centre', 'lat_centre', 'lon_mslp','lat_mslp','min_mslp','850vo','t63vo','925ws6','distance','track_id','orig_track_id']]
            all_cyclones_list.append(df_tr_new)

        
            # # Remove this track
            df_tracks_loop = df_tracks_loop.iloc[point_num:]
            new_track_id += 1
            
            # make list to numpy array
    all_cyclones = pd.concat(all_cyclones_list)
    
    return all_cyclones, number_of_tracks

def cluster_analysis(close_cyclones, slp_lim, radius):

    all_dates = pd.date_range(start='1979-09-28', end='2019-04-03', freq='1 D')

    intense_cyclones = pd.DataFrame(index=all_dates[3:-3], columns=['p95_day', 'p95_window','mean_intensity'])
    all_cyclones = pd.DataFrame(index=all_dates[3:-3], columns=['all_day', 'all_window','mean_intensity'])

    for d in all_dates[3:-3]:
        # print('Calculating clusters for day ' + d.strftime('%d-%m-%Y'))
        # define the 7-day window
        start_time = d - pd.Timedelta('3 days')
        end_time = d + pd.Timedelta('3 days') + pd.Timedelta('21 hours')
        ce_time_window = pd.date_range(start=start_time, end=end_time,freq='3H',)
        ce_time_day = pd.date_range(start=d, end=d+pd.Timedelta('21 hours'),freq='3H',)

    
        ## cyclones overlapping the window and day
        cyclones_window = close_cyclones[close_cyclones.index.isin(ce_time_window)]
        cyclones_day = close_cyclones[close_cyclones.index.isin(ce_time_day)]
    
        ## the timesteps when the cyclone is within 700 km radius
        cluster_cyclones_window = cyclones_window[cyclones_window.distance < radius]
        cluster_cyclones_day = cyclones_day[cyclones_day.distance < radius]
    
        # minimum pressures of the close cyclones
        pres_window = cluster_cyclones_window.min_mslp.groupby(cluster_cyclones_window.track_id).min()
        pres_day = cluster_cyclones_day.min_mslp.groupby(cluster_cyclones_day.track_id).min()
    
        # number of cyclones
        n_tracks = len(np.unique(cluster_cyclones_window.track_id))
        all_cyclones.loc[d]['all_window'] = n_tracks
        n_tracks = len(np.unique(cluster_cyclones_day.track_id))
        all_cyclones.loc[d]['all_day'] = n_tracks
    
        # number of intense cyclones
        intense_track_ids = pres_window<slp_lim
        intense_cyclones.loc[d]['p95_window'] = int(np.sum(intense_track_ids))
        
        intense_track_ids = pres_day<slp_lim
        intense_cyclones.loc[d]['p95_day'] = int(np.sum(intense_track_ids))
        
        intense_cyclones.loc[d]['mean_intensity'] = pres_window[pres_window<slp_lim].mean()

        
    return all_cyclones, intense_cyclones

def define_cyclone_time_series(close_cyclones, slp_lim, radius, year1, year2):
    
    all_dates = pd.date_range(start=str(year1-1)+'-09-28', end=str(year2)+'-04-03', freq='1 D')

    cyclone_df = pd.DataFrame(index=all_dates[3:-3], columns=['cyclone_day', 'mslp','mslp_mean','850vo','t63vo','925ws6','track_ids'])
    cyclone_df.cyclone_day = 0
    
    ids = np.unique(close_cyclones.track_id)
    
    for track_id in ids:
        
        # Whole track of the cyclone
        cyclone_whole_track = close_cyclones[close_cyclones.track_id==track_id]
        
        # Only that part of the track which is within the radius
        cyclone_part_track = cyclone_whole_track[cyclone_whole_track.distance<=radius]
        
        # if the minimum value is lower than the threshold
        if cyclone_part_track.min_mslp.min() <= slp_lim:  
                
            # day of relative minimum pressure (i.e. the day of highest intensity within the circle)
            day_of_mlsp_min = cyclone_part_track.min_mslp.idxmin().normalize()
            
            cyclone_df['cyclone_day'][day_of_mlsp_min] += 1
            
            # append the minimum mslp to the list if the value is list
            if isinstance(cyclone_df['mslp'][day_of_mlsp_min], list):
                cyclone_df['mslp'][day_of_mlsp_min].append(cyclone_part_track.min_mslp.min())
                cyclone_df['850vo'][day_of_mlsp_min].append(cyclone_part_track['850vo'].max())
                cyclone_df['t63vo'][day_of_mlsp_min].append(cyclone_part_track['t63vo'].max())
                cyclone_df['925ws6'][day_of_mlsp_min].append(cyclone_part_track['925ws6'].max())
            else:
                cyclone_df['mslp'][day_of_mlsp_min] = [cyclone_part_track.min_mslp.min()]
                cyclone_df['850vo'][day_of_mlsp_min] = [cyclone_part_track['850vo'].max()]
                cyclone_df['t63vo'][day_of_mlsp_min] = [cyclone_part_track['t63vo'].max()]
                cyclone_df['925ws6'][day_of_mlsp_min] = [cyclone_part_track['925ws6'].max()]
                
                
            # append the track id to the list if the value is list
            if isinstance(cyclone_df['track_ids'][day_of_mlsp_min], list):
                cyclone_df['track_ids'][day_of_mlsp_min].append(track_id)
            else:
                cyclone_df['track_ids'][day_of_mlsp_min] = [track_id]
    
            cyclone_df['mslp_mean'][day_of_mlsp_min] = np.mean(cyclone_df['mslp'][day_of_mlsp_min])
            
    clustering = cyclone_df.cyclone_day.rolling(window=7, center=True).sum()
    
    cyclone_df = pd.concat([cyclone_df, clustering.rename('7-day sum')], axis=1)
            
    return cyclone_df

        
def define_clustering_periods(cyclone_df):
    
    etc_df = cyclone_df.copy()

    # Define SCC3 events: the number of "intense" cyclones equals or is greater than 3
    scc3_all = etc_df['7-day sum'] >= 3
    scc3_periods = define_final_scc_periods(scc3_all, etc_df)

    # Define SCC2 events: the number of intense cyclones equals 2
    # scc2_all = cyclone_df['7-day sum'] == 2
    etc_df['cyclone_day'][scc3_periods] = 0
    scc2_all = etc_df.cyclone_day.rolling(window=7, center=True).sum() == 2
    scc2_periods = define_final_scc_periods(scc2_all, etc_df) 
    scc2_periods[scc3_periods]=False
    
    # Define single cyclone events: the number of intense cyclones is 1
    etc_df['cyclone_day'][scc2_periods] = 0
    single_cyclone_days = (etc_df.cyclone_day.rolling(window=7, center=True).sum() == 1) & \
                          (etc_df['cyclone_day'] == 1)
    # single_cyclone_days = (cyclone_df['7-day sum'] == 1) & (cyclone_df['cyclone_day'] == 1)   

    # Define single cyclone periods: add ± 1 days to each SC event
    single_cyclone_periods = define_single_periods(single_cyclone_days, etc_df)
    single_cyclone_periods[scc3_periods]=False
    single_cyclone_periods[scc2_periods]=False
    
    # merge the periods into one dataframe
    
    scc_df = pd.concat([scc3_periods.rename('SCC3'), scc2_periods.rename('SCC2'), single_cyclone_periods.rename('SCC1')], axis=1)
    
    return scc_df

    

def remove_short_scc_periods(scc_all, threshold):
    
    from scipy import ndimage
    
    # generate the structure to label each scc event
    struct = np.zeros(shape=(3))
    struct[:] = 1
    
    # label each scc event
    labels, nb = ndimage.label(scc_all, structure=struct)
    
    # calculate the length of each scc
    scc_lengths = np.array(ndimage.sum(scc_all, labels, np.arange(labels.max()+1)))
    
    # mask scc events which are shorther than three days
    mask = scc_lengths > threshold
    remove_small_scc = mask[labels.ravel()].reshape(labels.shape)
    
    # make labeled array
    scc_events = pd.Series(data=remove_small_scc, index=scc_all.index)
    
    # label each scc event
    labels, nb = ndimage.label(scc_events, structure=struct)
    scc_labels = pd.Series(data=labels, index=scc_all.index)
    
    return scc_lengths[1:], scc_labels

def define_final_scc_periods(scc_all, cyclone_df):
    
    dates = scc_all.index
    scc_periods = scc_all.copy(deep=True)
    scc_periods[:] = False
    
    for d in dates[3:-3]:
        
        # date_range = pd.date_range(start=d-pd.Timedelta('3 days'), end=d+pd.Timedelta('3 days'))
        
        # If the 7-day running sum is at least three
        if scc_all[d]:
            
            # Find cyclones from the previous 6 days
            date_range = pd.date_range(end=d+pd.Timedelta('3 days'), start=d-pd.Timedelta('3 days'))
            
            # indices of cyclone dates
            array = cyclone_df.loc[date_range].cyclone_day.to_numpy().nonzero()
            
            # if there are ETCs
            if np.size(array) > 0:
                
                # index of first ETC
                idx = array[0][0]
                # Date of first ETC
                date_first_etc = date_range[idx]
                
                # index of last ETC
                idx = array[0][-1]
                # Date of last ETC
                date_last_etc  = date_range[idx]
                
                # select dates from the first ETC to clustered day
                new_date_range =  pd.date_range(start=date_first_etc-pd.Timedelta('1 day'), 
                                                end=date_last_etc+pd.Timedelta('1 day'))
                # mark these days as SCC periods
                scc_periods[new_date_range] = True
        
    return scc_periods

def define_single_periods(single_cyclone_days, cyclone_df):
    
    dates = single_cyclone_days.index
    sc_periods = single_cyclone_days.copy(deep=True)
    sc_periods[:] = False

    
    for d in dates[3:-3]:
        
        date_range = pd.date_range(start=d-pd.Timedelta('1 days'), end=d+pd.Timedelta('1 days'))
        
        if (single_cyclone_days[d]) & all(cyclone_df['7-day sum'][date_range]==1):
            sc_periods[date_range] = True
        
    return sc_periods
        
def read_events(place,path, func):
    
    import matplotlib.dates as mdates
    
    if place=='pärnu' or place=='riga':
        datafile = path + '/' + place.capitalize() +  '_1961_2022.txt'
        df = pd.read_csv(datafile,delim_whitespace=True, header=None,names=['Year', 'Month','Day','Hour','Sea_level']) 
        df.index= pd.to_datetime(df[['Year','Month','Day','Hour']])
        df = df.drop(columns=['Year','Month','Day','Hour'])
        # mark missing values as nans
        df[df.Sea_level<=-900] = np.nan
        # subtract 500 to get values in N2000 system
        df = df-500
    elif place=='furuögrund' or place=='kungholmsfort':
        datafile = path + '/smhi-' + place +  '.csv'
        df = pd.read_csv(datafile, header=4,parse_dates=True, index_col=0, delimiter=';')     
        df = df.drop(columns=['Kvalitet','Mätdjup (m)'])
        df = df.rename(columns={'Havsvattenstånd':'Sea_level'})
    elif place=='landsort':
        datafile = path + '/smhi-' + place +  '.csv'
        df1 = pd.read_csv(datafile, header=4,parse_dates=True, index_col=0, delimiter=';')
        datafile = path + '/smhi-' + place +  '_norra.csv'
        df2 = pd.read_csv(datafile, header=4,parse_dates=True, index_col=0, delimiter=';') 
        df1 = df1.loc[slice('1900-01-01','2005-12-31')]
        df2 = df2.loc[slice('2006-01-01','2022-12-31')]
        df = pd.concat([df1, df2], axis=0)
        df = df.drop(columns=['Kvalitet','Mätdjup (m)'])
        df = df.rename(columns={'Havsvattenstånd':'Sea_level'}) 
    else:
        datafile = path + '/' + place.capitalize() + '_hourlystats_N2000.txt'
        df = pd.read_csv(datafile,sep=" ", header=None, index_col=1, parse_dates=True, names=['ID', 'Sea_level','flag'])
        # convert to centimeters
        df.Sea_level = df.Sea_level/10


    df.index.name='time'
    
    df = df[(df.index.year>=1979) & (df.index.year<=2022)]
    # df = df[(df.index.month<=3)|(df.index.month>=10)]
    
    all_times = pd.date_range('1979-01-01','2022-12-31T23:00', freq='H')
    df = df.reindex(all_times)
    

    
    
    dailymeans = df[(df.index.year>=1979) & (df.index.year<=2022)].groupby(pd.Grouper(freq='1D')).mean()
    
    # calculate annual means
    yearlymeans = dailymeans.groupby(pd.Grouper(freq='1Y')).mean()
    
    if func=='max':
        df = df.groupby(pd.Grouper(freq='D')).max()
    elif func=='min':
        df = df.groupby(pd.Grouper(freq='D')).min()
    elif func=='mean':
        df = df.groupby(pd.Grouper(freq='D')).mean()
    elif func=='all':
        df = df
    
    idx = np.sum(np.isnan(df.Sea_level))
    
    idx_missing = np.round((idx / len(df))*100,2)
    
    print('Missing values: ', str(idx_missing),'%')
        
    
    ### calculate linear trend in yearly means
    y = np.array(yearlymeans.Sea_level.values)
    x = mdates.date2num(yearlymeans.index.values)
    idx = ~np.isnan(y)

    z = np.polyfit(x[idx], y[idx], 1)
    p2 = np.poly1d(z)
    xx = np.linspace(x.min(), x.max(), len(df))
    
    ### remove the *yearly* trend from daily maximum values
    df.Sea_level = df.Sea_level - p2(xx)
    
    # print the magnitude of the trend that was removed
    print(place, 'Trend in yearly means: '+str(np.round(z[0]*365*10,1))+ ' cm per decade')
    
    return df

def define_extreme_sl_events(df, threshold):
    
    sl = (df>threshold)*1
    
    # sl_window = sl.rolling(window=7, center=True).sum()
    
    sl_events = df.copy(deep=True)
    sl_events[:] = 0
    
    dates = df.index
    
    for d in dates[3:-3]:
        
        # date_range = pd.date_range(start=d-pd.Timedelta('3 days'), end=d+pd.Timedelta('3 days'))
        
        # If the 7-day running sum is at least three
        if sl[d]>=1:
            
            # Find sl events from the next 6 days
            date_range = pd.date_range(end=d+pd.Timedelta('6 days'), start=d)
            
            # indices of cyclone dates
            array = sl.loc[date_range].to_numpy().nonzero()
            
            # if there are ETCs
            if np.size(array) > 0:
                
                # index of first ETC
                idx = array[0][0]
                # Date of first ETC
                date_first_etc = date_range[idx]
                
                # index of last ETC
                idx = array[0][-1]
                # Date of last ETC
                date_last_etc  = date_range[idx]
                
                # select dates from the first ETC to clustered day
                new_date_range =  pd.date_range(start=date_first_etc-pd.Timedelta('0 day'), 
                                                end=date_last_etc+pd.Timedelta('0 day'))
                # mark these days as SCC periods
                sl_events[new_date_range] = 1
    
    return sl_events[(sl_events.index.month<=3)|(sl_events.index.month>=10)]

def define_clustering_ids(track_ids, scc_lengths, scc_onset_dates):
    
    clustering_ids = dict.fromkeys(scc_onset_dates.index)
    
    for i,sccdate in enumerate(scc_onset_dates.index):
        
        lenght = scc_lengths[i]
        
        dates = pd.date_range(start=sccdate, periods=lenght, freq='D')
        
        ids = list(track_ids.loc[dates].values)
        cleanedList = [x for x in ids if str(x) != 'nan']
        flat_ids = [item for sublist in cleanedList for item in sublist]
    
        clustering_ids[sccdate] = flat_ids
        
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in clustering_ids.items() ])).transpose()
        
    return df

def define_clustering_mslps(mslps, scc_lengths, scc_onset_dates):
    
    clustering_ids = dict.fromkeys(scc_onset_dates.index)
    
    for i,sccdate in enumerate(scc_onset_dates.index):
        
        lenght = scc_lengths[i]
        
        dates = pd.date_range(start=sccdate, periods=lenght, freq='D')
        
        msls = list(mslps.loc[dates].values)
        cleanedList = [x for x in msls if str(x) != 'nan']
        flat_ids = [item for sublist in cleanedList for item in sublist]
    
        clustering_ids[sccdate] = flat_ids
        
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in clustering_ids.items() ])).transpose()
        
    return df

def define_maximum_sealevel(df, sl_onset_dates, sl_lenghts):
    
    max_sealevel = pd.DataFrame(index=sl_onset_dates.index, columns=['Max sea level'])
    
    for i,sdate in enumerate(sl_onset_dates.index):
        
        lenght = sl_lenghts[i]
    
        dates = pd.date_range(start=sdate, periods=lenght, freq='D')
        
        slmax = df.loc[dates].max()
        
        max_sealevel.loc[sdate] = slmax
    
    return max_sealevel

def calculate_sealevel_climatology(N, scc_dates, sea_level, lag):
    
    import random
    
    # read sea level data
    # sea_level = read_events(place, sl_path, func='max').Sea_level
    
    # allocate dataframe to the bootstrapping results
    scc_climatology = pd.DataFrame(columns=np.arange(0,N), index=np.arange(-lag,lag+1))
    
    # climatology years
    years = np.arange(1980,2023)
    
    # make the repetitions
    for i in np.arange(0,N):
        
        
        scc_climatology_df = pd.DataFrame(columns=scc_dates, index=np.arange(-lag,lag+1))
        
        random_dates_list = []
        
        for scc_date in scc_dates:
        
        
            current_year = scc_date.year
            
            if scc_date.month>=9:
                current_year +=1
            
            
            # Construct dates +- 15 days around the SCC onset date
            dates = pd.date_range(start=scc_date-pd.Timedelta('15 days'), 
                                  end=scc_date+pd.Timedelta('15 days'))
            
            # remove the leap day
            leapDays = (dates.month==2) & (dates.day==29)
            dates = dates[~leapDays]
            
            # select randomly a year from 1980-2019 (excluding the current year)
            random_year = random.choice(np.setdiff1d(years,current_year))
            
            # select randomly a calendar day +/- 15 days around the actual calendar day
            random_date = random.choice(dates).replace(year=random_year)
            if random_date.month>=9:
                random_date = random_date-pd.DateOffset(years=1)
            
            # Construct the lagged climatology dates +- 30 days around the random date
            clim_dates = pd.date_range(start=random_date-pd.Timedelta(lag, 'D'), 
                                  end=random_date+pd.Timedelta(lag, 'D'))
            
            
            
            # read sea level data from those random dates
            sl_random = sea_level.loc[clim_dates].rolling(window=1, center=True).mean()
            
            
            random_dates_list.append(random_date)
            

            scc_climatology_df[scc_date] = sl_random.values
            
            
        scc_climatology[i] = scc_climatology_df.mean(axis=1)
        
    return scc_climatology
        
def calculate_sealevel_climatology_hourly(N, scc_dates, sea_level, lag):
    
    import random
    
    # read sea level data
    # sea_level = read_events(place, sl_path, func='max').Sea_level
    
    # allocate dataframe to the bootstrapping results
    scc_climatology = pd.DataFrame(columns=np.arange(0,N), index=np.arange(-lag,lag+1))
    
    # climatology years
    years = np.arange(1980,2023)
    
    # make the repetitions
    for i in np.arange(0,N):
        
        
        scc_climatology_df = pd.DataFrame(columns=scc_dates, index=np.arange(-lag,lag+1))
        
        random_dates_list = []
        
        for scc_date in scc_dates:
        
        
            current_year = scc_date.year
            
            if scc_date.month>=9:
                current_year +=1
            
            
            # Construct dates +- 15 days around the SCC onset date
            dates = pd.date_range(start=scc_date-pd.Timedelta('15 days'), 
                                  end=scc_date+pd.Timedelta('15 days'), freq='1H')
            
            # remove the leap day
            leapDays = (dates.month==2) & (dates.day==29)
            dates = dates[~leapDays]
            
            # select randomly a year from 1980-2019 (excluding the current year)
            random_year = random.choice(np.setdiff1d(years,current_year))
            
            # select randomly a calendar day +/- 15 days around the actual calendar day
            random_date = random.choice(dates).replace(year=random_year)
            if random_date.month>=9:
                random_date = random_date-pd.DateOffset(years=1)
            
            # Construct the lagged climatology dates +- 30 days around the random date
            clim_dates = pd.date_range(start=random_date-pd.Timedelta(lag, 'D'), 
                                  end=random_date+pd.Timedelta(lag, 'D'))
            
            
            
            # read sea level data from those random dates
            sl_random = sea_level.loc[clim_dates].rolling(window=1, center=True).mean()
            
            
            random_dates_list.append(random_date)
            

            scc_climatology_df[scc_date] = sl_random.values
            
            
        scc_climatology[i] = scc_climatology_df.mean(axis=1)
        
    return scc_climatology 

def calculate_cluster_intensities(intensity, scc_lengths, scc_onset_dates):
    
    
    scc_mslps = define_clustering_ids(intensity, scc_lengths, scc_onset_dates)
    scc = pd.concat([scc_lengths.rename('duration'), scc_mslps], axis=1)
    
    
    return scc
