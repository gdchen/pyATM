#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:42:43 2022

@author: yaoyichen
"""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta,time
from mpl_toolkits.basemap import Basemap
import os
from os import walk


# def TimeStampToTime(timestamp):
#     timeStruct = time.localtime(timestamp)
#     return time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)


# def get_FileModifyTime(filePath):
#     t = os.path.getmtime(filePath)
#     return TimeStampToTime(t)


def plus_seconds(x):
    start_time = datetime(1970, 1, 1, 0, 0, 0)
    result_time = start_time + timedelta(seconds=x)
    return result_time


def find_datetime_within( time_vector, value_vector, start_time, end_time ):
    """
    找到符合条件的index
    """
    # index_list = []
    time_list = []
    value_list = []
    for time_value,value_value in  zip(time_vector,value_vector):
        if ((time_value >= start_time) and (time_value <= end_time)):
            # index_list.append(index)
            time_list.append(time_value)
            value_list.append(value_value)
    return [time_list,value_list]


def get_filedetail_from_folder(folder_name, endswith_str = None, startswith_str = None):
    """
    获取 folder_name 中所有的有效的文件名,按endswith_str 和 startswith_str 过滤
    
    """

    folder_file_list = [] 
    for (dirpath, dirnames, filenames) in walk(folder_name):
        
        for filename in filenames:
            if(endswith_str and startswith_str):
                if(filename.endswith(endswith_str) and filename.startswith(startswith_str)):
                    folder_file_list.append(os.path.join(dirpath, filename))
            
            elif(endswith_str):
                if(filename.endswith(endswith_str) ):
                    folder_file_list.append(os.path.join(dirpath, filename))

            elif(startswith_str):
                if(filename.startswith(startswith_str) ):
                    folder_file_list.append(os.path.join(dirpath, filename))
            else:
                folder_file_list.append(os.path.join(dirpath, filename))
    

    
    return folder_file_list


file_name = "/Users/yaoyichen/dataset/carbon/ObsPack/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18/data/daily/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20201231.nc"
df = nc.Dataset(file_name)

longitude = np.asarray(df.variables["longitude"][:])
latitude = np.asarray(df.variables["latitude"][:])
time_original = np.asarray(df.variables["time"][:])
time = [plus_seconds(int(x)) for x in time_original]
"""
测量距离地面的高度 
"""
intake_height = np.asarray(df.variables["intake_height"][:])

model_sample_window_start = np.asarray(df.variables["model_sample_window_start"][:])
model_sample_window_end =  np.asarray(df.variables["model_sample_window_end"][:])
start_time = [plus_seconds(int(x)) for x in model_sample_window_start]
end_time  = [plus_seconds(int(x)) for x in model_sample_window_end]

value  = np.asarray(df.variables["value"][:])



#%%
folder_name = "/Users/yaoyichen/dataset/carbon/ObsPack/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18_site/data/nc/"
file_name_list = ["co2_smr_surface-insitu_442_allvalid-17magl.nc"]


file_name_list = get_filedetail_from_folder(folder_name,endswith_str = ".nc")
# file_name = os.path.join(folder_name,file_name_list[0])
# for file_name in file_name_list:
    
for file_name in file_name_list:
    df = nc.Dataset(os.path.join(folder_name,file_name))
    
    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    time_original = np.asarray(df.variables["time"][:])
    
    
    time_vector = [plus_seconds(int(x)) for x in time_original]
    value_vector  = list(np.asarray(df.variables["value"][:]))
    # value_std  = np.asarray(df.variables["value_std_dev"][:])    
    
    start_time = datetime(2019, 7, 1, 0, 0)
    end_time  = datetime(2019, 7, 8, 0, 0)
    [time_list, value_list] = find_datetime_within( time_vector, value_vector, start_time, end_time )
    
    #%%
    
    
    fig = plt.figure(tight_layout=True, figsize=(10,4))
    gs = gridspec.GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    
    if(len(time_list) <= 30 ):
        continue
    
    plot_string = f"{df.site_code};{df.site_country}; {df.site_name}; {df.site_longitude} ;{df.site_latitude}; {df.site_elevation}"
    ax.plot(time_list, value_list)
    ax.set_title(plot_string)
    ax.grid()
    x_ticks = ax.get_xticks()
    myFmt = mdates.DateFormatter('%Y-%m-%d')
    
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    ax.xaxis.set_major_formatter(myFmt)
    #%%
    
    ax = fig.add_subplot(gs[0, 1])
    m = Basemap(projection='merc', \
                llcrnrlat=-70, urcrnrlat=70, \
                llcrnrlon=-180, urcrnrlon=180, \
                lat_ts=20, \
                resolution='c' )
    
    m.bluemarble(scale=0.2)   # full scale will be overkill
    m.drawcoastlines(color='white', linewidth=0.2)  # add coastlines
    
    lons = [df.site_longitude]
    lats = [df.site_latitude]
        
    x, y = m(lons, lats)  # transform coordinates
    
    ax.scatter(x, y, s = 20, marker='o', color='Red') 
    
    plot_filename = f"./plot_folder/plot_{plot_string}.png"
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    plt.cla()


