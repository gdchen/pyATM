#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:09:14 2022

@author: yaoyichen
"""
from data_prepare.generate_observation_data import generate_obspack_data,generate_satellite_data
import os
from data_prepare.plot_utils import plot_lonlat_onearth
import numpy as np

year = 2019
month = 7
day = 1
last_day = 3
interval_minutes = 30
data_folder = '/Users/yaoyichen/dataset/auto_experiment/experiment_0/'
result_folder = '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/tt_0422/'


satellite_folder = os.path.join(data_folder,  'satellite')

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

[index_vector, satellite_value_vector1, obs_time_vector1, obs_longitude_vector1, obs_latitude_vector1] = \
    generate_satellite_data(year, month, 1, 
                          last_day, interval_minutes*60,
                satellite_folder, result_folder, f"satellite_{year}{str(month).zfill(2)}{str(day).zfill(2)}_{str(last_day).zfill(2)}")

#%%
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.dates as mdates

target_long = -95
target_lati = 34
time_list = []
result_list = []
start_time = datetime.datetime(2019, 7, 1, 0, 0)


for (time_vector1, longitude_index1, latitude_index1,value) in zip(obs_time_vector1, obs_longitude_vector1, obs_latitude_vector1,satellite_value_vector1.numpy()):
    if((longitude_index1 == target_long) and (latitude_index1 == target_lati)):
        result_list.append(value)
        time_list.append(start_time + datetime.timedelta(hours = int(time_vector1)//2))


fig = plt.figure(tight_layout=True, figsize=(10,4))
gs = gridspec.GridSpec(1, 2)
ax = fig.add_subplot(gs[0, 0])

ax.scatter(time_list, result_list)
myFmt = mdates.DateFormatter('%Y-%m-%d')
    
ax.set_xticklabels(ax.get_xticks(), rotation = 45)
ax.xaxis.set_major_formatter(myFmt)

#%%

plot_lonlat_onearth(obs_longitude_vector,obs_latitude_vector,
                    satellite_value_vector.numpy(), dot_size = 20 )


[index_vector, satellite_value_vector2, obs_time_vector2, obs_longitude_vector2, obs_latitude_vector2] = \
    generate_satellite_data(year, month, 15, 
                          last_day, interval_minutes*60,
                satellite_folder, result_folder, f"satellite_{year}{str(month).zfill(2)}{str(day).zfill(2)}_{str(last_day).zfill(2)}")


plot_lonlat_onearth(obs_longitude_vector2, obs_latitude_vector2,
                    satellite_value_vector2.numpy(), dot_size = 20 )

#%%
for (time_vector1, longitude_index1, latitude_index1) in zip(obs_time_vector1, obs_longitude_vector1, obs_latitude_vector1):
    print(time_vector1, longitude_index1, latitude_index1)
    
for (time_vector2, longitude_index2, latitude_index2) in zip(obs_time_vector2, obs_longitude_vector2, obs_latitude_vector2):
    print(time_vector2, longitude_index2, latitude_index2)

#%%

input_list = []
for (longitude_index1, latitude_index1) in zip(obs_longitude_vector1, obs_latitude_vector1):
    input_str1 = str(longitude_index1).zfill(4) + str(latitude_index1).zfill(4)
    
    for (longitude_index2 ,latitude_index2) in zip(obs_longitude_vector2, obs_latitude_vector2):
        input_str2 = str(longitude_index2).zfill(4) + str(latitude_index2).zfill(4)
        
        if(input_str1 == input_str2):
            input_list.append(input_str1)
            break

#%%
value1_list = []
value2_list = []

for input_ in input_list:
    longitude_index_query = int(input_[0:4])
    latitude_index_query  = int(input_[4:8])
    
    for (longitude_index1, latitude_index1, satellite_value1) in zip(obs_longitude_vector1, obs_latitude_vector1, satellite_value_vector1.numpy()):
        if((longitude_index_query == longitude_index1) and (latitude_index_query == latitude_index1)):
            value1_list.append(satellite_value1)

    for (longitude_index2, latitude_index2, satellite_value2) in zip(obs_longitude_vector2, obs_latitude_vector2, satellite_value_vector2.numpy()):
        if((longitude_index_query == longitude_index2) and (latitude_index_query == latitude_index2)):
            value2_list.append(satellite_value2)
        
    
#%%



