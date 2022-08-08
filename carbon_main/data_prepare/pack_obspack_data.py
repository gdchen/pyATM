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
from plot_utils import plot_lonlat_onearth
from output_utils import assign2grid, averaging_obs, average_duplicatd_obs

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


file_list = ["obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20190701.nc",
             "obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20190702.nc",
             "obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20190703.nc",
             "obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20190704.nc",
             "obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20190705.nc"]


value_list = []
time_list = []
longitude_list = []
latitude_list = []

for file_ in file_list:
# file_ = file_list[0
    
    file_name = os.path.join("/Users/yaoyichen/dataset/carbon/ObsPack/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18/data/daily/",
                             file_)
    df = nc.Dataset(file_name)
    
    
    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    time_original = np.asarray(df.variables["time"][:])
    time = [plus_seconds(int(x)) for x in time_original]
    """
    测量距离地面的高度 
    """
    intake_height = np.asarray(df.variables["intake_height"][:])
    
    valid_index = (intake_height<1000) & (intake_height >1)
    
    model_sample_window_start = np.asarray(df.variables["model_sample_window_start"][:])
    model_sample_window_end =  np.asarray(df.variables["model_sample_window_end"][:])
    start_time = [plus_seconds(int(x)) for x in model_sample_window_start]
    end_time  = [plus_seconds(int(x)) for x in model_sample_window_end]
    
    value  = np.asarray(df.variables["value"][:])


    value_list += list(value[valid_index])
    for i in range(len(valid_index)):
        if(valid_index[i] == True):
            # print(time, time[i])
            time_list += [time[i]]
    longitude_list += list(longitude[valid_index])
    latitude_list += list(latitude[valid_index])
    
longitude_vector = np.asarray(longitude_list)
latitude_vector = np.asarray(latitude_list)
value_vector = np.asarray(value_list)

# plot_lonlat_onearth(longitude_vector, latitude_vector,value_vector )
# plot_lonlat_onearth(longitude,latitude, xco2 )


longitude_vector,latitude_vector,time_vector = assign2grid( 
    latitude_spacing = 4,
    longitude_spacing = 5,
    time_spacing = 3600/2,
    start_time =  datetime(2019,7,1,0,0,0),
    latitude = latitude_vector,
    longitude = longitude_vector,
    time_vector = time_list)


value_vector,longitude_vector,latitude_vector, time_vector,result_dict =  \
average_duplicatd_obs(longitude_vector,latitude_vector,time_vector, value_vector,  
            map_shape = (240,72,46),max_clip = None, min_clip = None)

plt.figure(1)
plot_lonlat_onearth(longitude_vector*5-180,latitude_vector*4-90, value_vector, dot_size = 30 )


#%%



# 筛一下时间
max_time_value = 240
extract_list = np.where(time_vector<max_time_value)[0]
value_vector = value_vector[extract_list]
longitude_vector = longitude_vector[extract_list]
latitude_vector = latitude_vector[extract_list]
time_vector = time_vector[extract_list]


#%%

index_list = []
for time_value, longitude_value, latitude_value in zip(time_vector, longitude_vector, latitude_vector):
    """
    np.ravel_multi_index -> 得到一维向量编码 
    np.unravel_index     -> 得到多维向量编码  
    """
    index_result = np.ravel_multi_index((time_value,longitude_value, latitude_value),  (240,72,46))
    index_list.append(index_result)
    


index_vector = np.asarray(index_list)
print(len(index_vector))

np.savez('obspack_data_20190701_240_72_46.npz', 
         index_vector = index_vector, 
         value_vector = value_vector,
         time_vector = time_vector)
