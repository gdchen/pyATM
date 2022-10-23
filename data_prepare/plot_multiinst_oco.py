#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:29:36 2022

@author: yaoyichen
"""

import h5py
import matplotlib.pyplot as plt
import  datetime
from mpl_toolkits.basemap import Basemap
import os
import numpy as np
from plot_utils import plot_lonlat_onearth
from read_utils import convertTime,plus_seconds
from output_utils import assign2grid, averaging_obs, average_duplicatd_obs


# folder_name = "/Users/yaoyichen/dataset/carbon/download OCO/OCO2_20190701_20190707/"
# file_name = "MultiInstrumentFusedXCO2_20190701_v2_1605481973.nc"

# foler_file_name = os.path.join(folder_name , file_name)
# f = h5py.File(foler_file_name, 'r')


# folder_name = "/Users/yaoyichen/dataset/carbon/download OCO/OCO2_20190701_20190707/MultiInstrumentFused/"
folder_name = "/Users/yaoyichen/Desktop/satellite_plot/"
# folder_name = "/Users/yaoyichen/dataset/auto_experiment/experiment_0/satellite"
file_list = os.listdir(folder_name)


def clean_file_list(file_list,startstring = None, endsstring = None):
    file_list_new = []
    for file_name in file_list:
        if(endsstring):
            if(not file_name.endswith(endsstring)):
                continue
        if(startstring):
            if(not file_name.startswith(startstring)):
                continue
        file_list_new.append(file_name)
        
    return file_list_new


file_list = clean_file_list(file_list,endsstring = ".nc")

def parse_from_file_list(folder_name, file_list):
    xco2_stack = np.array([])
    xco2_uncertainty_stack = np.array([])
    latitude_stack = np.array([])
    longitude_stack = np.array([])
    time_stack = []
    
    for file_name in file_list:
        foler_file_name = os.path.join(folder_name , file_name)
        f = h5py.File(foler_file_name, 'r')
        
        xco2 = f["xco2"][:]
        xco2_uncertainty = f["xco2_uncertainty"][:]
        latitude  = f["latitude"][:]
        longitude  = f["longitude"][:]
        
        time_original = f["time"][:]
        time = [plus_seconds(int(x)) for x in time_original]
    
        print(latitude.shape)
        latitude_stack = np.hstack([latitude_stack,latitude])
        longitude_stack = np.hstack([longitude_stack,longitude])
        xco2_stack = np.hstack([xco2_stack, xco2])
        xco2_uncertainty_stack = np.hstack([xco2_uncertainty_stack, xco2_uncertainty])
        time_stack += time
    
    return [xco2_stack, xco2_uncertainty_stack, latitude_stack, longitude_stack, time_stack]


file_list.sort()
# for sub_file_list in file_list:
#     print(sub_file_list)
xco2, xco2_uncertainty_stack, latitude, longitude, time = parse_from_file_list(folder_name,file_list)
    
    
# 画图代码
plot_lonlat_onearth(longitude,latitude, xco2 )
# plot_lonlat_onearth(longitude,latitude, xco2_uncertainty_stack )

#%%

longitude_vector,latitude_vector,time_vector = assign2grid( 
    latitude_spacing = 4,
    longitude_spacing = 5,
    time_spacing = 3600/2,
    start_time =  datetime.datetime(2019,7,1,0,0,0),
    latitude = latitude,
    longitude = longitude,
    time_vector = time)


value_vector,longitude_vector,latitude_vector, time_vector,result_dict =  \
average_duplicatd_obs(longitude_vector,latitude_vector,time_vector, xco2,  
            map_shape = (240,72,46),max_clip = None, min_clip = None)

plt.figure(1)
plot_lonlat_onearth(longitude_vector*5-180,latitude_vector*4-90, value_vector, dot_size = 30 )
#%%latitude_vector

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

np.savez('satellite_data_20190701_240_72_46.npz', 
         index_vector = index_vector, 
         value_vector = value_vector,
         time_vector = time_vector)

#%%
    



#%%
# def subplot(result_dict):
#     total_longitude_list = []
#     total_latitude_list = []
#     total_value_list = []
#     for i in range(168):
#         print(i)
#         longitude_list, latitude_list, value_list = [],[] , []
#         for key,value in result_dict.items():
#             temp = key.split("_")
#             longtitude = int(temp[0])
#             latitude = int(temp[1])
#             time_ = int(temp[2])
#             if(time_ == i):
#                 longitude_list.append(longtitude)
#                 latitude_list.append(latitude)
#                 value_list.append(value["value"])
#         total_longitude_list += longitude_list
#         total_latitude_list += latitude_list
#         total_value_list += value_list
            
            
#         if(i %10 ==0):
#             print(len(longitude_list), len(latitude_list), len(value_list) )
#             plt.figure(i)
#             plt.scatter(total_longitude_list, total_latitude_list, c= total_value_list,s = 5, vmax = 412, vmin =405)

# subplot(result_dict)

