#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:13:25 2022

@author: yaoyichen
"""

import h5py
import matplotlib.pyplot as plt
import  datetime
from mpl_toolkits.basemap import Basemap
import os
import numpy as np
from plot_utils import plot_lonlat_onearth
from read_utils import convertTime

"""
h5 file group https://docs.h5py.org/en/stable/high/group.html
how to read from h5py group https://stackoverflow.com/questions/56184984/read-multiple-datasets-from-same-group-in-h5-file-using-h5py

file format: oco2_L2Std[Mode]_[Orbit][ModeCounter]_[AcquisitionDate]_[ShortBuildId]_[Production DateTime] [Source].h5
"""




def read_from_oco_data(folder_name, file_list):
    """
    OCO-3 Level 2 geolocated XCO2 retrievals results, physical model, Forward Processing V10 (OCO3_L2_Standard)
    地址 https://disc.gsfc.nasa.gov/datasets/OCO3_L2_Standard_10/summary?keywords=OCO-3



    Parameters
    ----------
    folder_name : TYPE
        DESCRIPTION.
    file_list : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    xco2_stack = np.array([])
    latitude_stack = np.array([])
    longitude_stack = np.array([])
    time_stack = []
    
    co2_profile_stack = np.array([[]]).reshape(0,20)
    vector_altitude_stack = np.array([[]]).reshape(0,20)
    vector_pressure_stack = np.array([[]]).reshape(0,20)
    time_stack = []
    for file_name in file_list:
        if(not file_name.endswith(".h5")):
            continue
        print(file_name)
        foler_file_name = os.path.join(folder_name , file_name)
        f = h5py.File(foler_file_name, 'r')
        # print( list(f.keys()) )
        
        xco2 = f["RetrievalResults"]["xco2"][:]
        co2_profile = f["RetrievalResults"]["co2_profile"][:]
        latitude = f["RetrievalGeometry"]["retrieval_latitude"][:]
        longitude = f["RetrievalGeometry"]["retrieval_longitude"][:]
        time_original = f["RetrievalHeader"]["retrieval_time_string"][:]
        time = [convertTime(x) for x in time_original]
        vector_altitude = f["RetrievalResults"]["vector_altitude_levels"][:]
        vector_pressure = f["RetrievalResults"]["vector_pressure_levels"][:]

        xco2_stack = np.hstack([xco2_stack,xco2])
        latitude_stack = np.hstack([latitude_stack,latitude])
        longitude_stack = np.hstack([longitude_stack,longitude])
        time_stack += time
        
        co2_profile_stack = np.vstack([co2_profile_stack, co2_profile])
        
        vector_altitude_stack = np.vstack([vector_altitude_stack, vector_altitude])
        vector_pressure_stack = np.vstack([vector_pressure_stack, vector_pressure])

    
    return [xco2_stack, co2_profile_stack, latitude_stack, longitude_stack, time_stack,vector_altitude_stack, vector_pressure_stack]


#%%


folder_name = "/Users/yaoyichen/dataset/carbon/OCO/"
file_list = ["oco3_L2StdSC_15083a_220102_B10310_220102234921.h5"]


folder_name = "/Users/yaoyichen/dataset/carbon/download OCO/data/"

file_list = os.listdir(folder_name)

    
[xco2, co2_profile, latitude, longitude, time_vector, vector_altitude, vector_pressure] = read_from_oco_data(folder_name, file_list)


#%%
if(True):
    plot_lonlat_onearth(longitude, latitude, xco2)



latitude_spacing = 2
longitude_spacing = 2
latitude_grid = 180//latitude_spacing + 1
longitude = 360 // longitude_spacing


latitude_map = {}
for i in range(91):
    latitude_map[2*i - 90] = i

longitude_map = {}
for i in range(180):
    longitude_map[2*i] = i

def modify_oco_latitude(latitude, latitude_map):
    result = []
    for value in latitude:
        index= latitude_map[2*int((value + 1)//2)]
        result.append(index)
    return np.asarray(result)


def modify_oco_longitude(longtitude, longitude_map):
    result = []
    for value in longtitude:
        if(value < 0):
            value += 360
        index= longitude_map[2*int((value)//2)]
        result.append(index)
    return np.asarray(result)

def modify_oco_time(time_vector, start_time):
    result = []
    for time_value in time_vector:
        time_delta = time_value - start_time
        hour_value = int(time_delta.total_seconds()//3600)
        result.append(hour_value)

    return np.asarray(result)
    

latitude_vector = modify_oco_latitude(latitude, latitude_map)
longitude_vector = modify_oco_longitude(longitude, longitude_map)
start_time = datetime.datetime(2022,1,1,0,0,0)
hour_vector = modify_oco_time(time_vector, start_time)

#%%

"""
数据去重，如果重复则设置为平均值
"""
result_dict = {}
for longitude, latitude, hour,value in zip(longitude_vector, latitude_vector, hour_vector,xco2):
    key = "_".join([str(longitude), str(latitude), str(hour)])
    if(not key in result_dict):
        result_dict[key] = {"value":value, "count":1}
    else:
        original_count= result_dict[key]["count"]
        original_value= result_dict[key]["value"]
        
        result_count = original_count+ 1
        result_value = 1.0*(original_count*original_value + value)/result_count
        result_dict[key] = {"value":result_value, "count":result_count}

#%%

index_list, value_list = [],[]
for key,value in result_dict.items():
    temp = key.split("_")
    # index_list.append([int(temp[0]), int(temp[1]), int(temp[2])])
    index_result = np.ravel_multi_index((int(temp[2]),int(temp[0]), int(temp[1])), (168,180,91))
    index_list.append(index_result)
    value_list.append(max(min(0.000425,value['value']),0.000395))

index_vector = np.asarray(index_list)
value_vector = np.asarray(value_list)

#%%
    

np.savez('oco_data_2022_first_week.npz', 
         index_vector = index_vector, 
         value_vector = value_vector)

# data = np.load('mat.npz')
# print data['name1']
# print data['name2']


#%%

def subplot():
    total_longitude_list = []
    total_latitude_list = []
    total_value_list = []
    for i in range(168):
        print(i)
        longitude_list, latitude_list, value_list = [],[] , []
        for key,value in result_dict.items():
            temp = key.split("_")
            longtitude = int(temp[0])
            latitude = int(temp[1])
            time_ = int(temp[2])
            if(time_ == i):
                longitude_list.append(longtitude)
                latitude_list.append(latitude)
                value_list.append(value["value"])
        total_longitude_list += longitude_list
        total_latitude_list += latitude_list
        total_value_list += value_list
            
            
        if(i %10 ==0):
            print(len(longitude_list), len(latitude_list), len(value_list) )
            plt.figure(i)
            plt.scatter(total_longitude_list, total_latitude_list, c= total_value_list,s = 5, vmax = 420e-6, vmin = 390e-6)


subplot()




# if __name__ = "__main__":
    