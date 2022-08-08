#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:29:36 2022

@author: yaoyichen
"""
import sys 
sys.path.append("..") 
import h5py
import matplotlib.pyplot as plt
import  datetime
# from mpl_toolkits.basemap import Basemap
import os
import numpy as np

from data_prepare.plot_utils import plot_lonlat_onearth
from data_prepare.read_utils import convertTime,plus_seconds
from data_prepare.output_utils import assign2grid, averaging_obs, average_duplicatd_obs
import netCDF4 as nc
import torch


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
    
        # print(latitude.shape)
        latitude_stack = np.hstack([latitude_stack,latitude])
        longitude_stack = np.hstack([longitude_stack,longitude])
        xco2_stack = np.hstack([xco2_stack, xco2])
        xco2_uncertainty_stack = np.hstack([xco2_uncertainty_stack, xco2_uncertainty])
        time_stack += time
    
    return [xco2_stack, xco2_uncertainty_stack, latitude_stack, longitude_stack, time_stack]


#%%


def generate_satellite_data(year, month, day, 
                            total_day_number, time_spacing,
                            data_folder_name, result_folder_name, output_file_name):

    
    start_datetime =  datetime.datetime(year, month, day,0,0,0)
    file_list = os.listdir(data_folder_name)
    file_list = clean_file_list(file_list,endsstring = ".nc")
    
    time_length = int(total_day_number*86400/time_spacing)
    
    selected_file_list = []
    for step_ in range ( total_day_number):
        step_datetime = start_datetime + datetime.timedelta(days = step_)
        str_datetime = step_datetime.strftime("%Y%m%d")
        # print(str_datetime)
        if_found = 0
        for file_str in file_list:
            if(str_datetime in file_str):
                print(f"finding satellite date:{str_datetime}")
                selected_file_list.append(file_str)
                if_found = 1
                break 
        if(if_found == 0):
            print(f"missing satellite date:{str_datetime}")
    
    xco2, xco2_uncertainty_stack, latitude_vector, longitude_vector, datetime_list \
    = parse_from_file_list(data_folder_name, selected_file_list)
    
    
    longitude_vector,latitude_vector,time_vector = assign2grid( 
        latitude_spacing = 4,
        longitude_spacing = 5,
        time_spacing = time_spacing,
        start_time =  start_datetime,
        latitude = latitude_vector,
        longitude = longitude_vector,
        time_vector = datetime_list)
    
    #%%
    value_vector, longitude_vector, latitude_vector, time_vector,result_dict =  \
    average_duplicatd_obs(longitude_vector,latitude_vector,time_vector, xco2,  
                map_shape = (time_length,72,46),max_clip = None, min_clip = None)
    
    # plt.figure(1)
    # plot_lonlat_onearth(longitude_vector*5-180,latitude_vector*4-90, value_vector, dot_size = 30 )
    #%%latitude_vector
    
    # 筛一下时间
    max_time_value = time_length
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
        index_result = np.ravel_multi_index((time_value,longitude_value, latitude_value),  (time_length,72,46))
        index_list.append(index_result)
    
    index_vector = np.asarray(index_list)
    print(f"total observation number:{len(index_vector)}")
    
    np.savez(os.path.join(result_folder_name, output_file_name), 
              index_vector = index_vector, 
              value_vector = value_vector,
              time_vector = time_vector)
    
    longitude_vector = longitude_vector*5 - 180 
    latitude_vector = latitude_vector*4 - 90
    
    index_vector_torch = torch.tensor(index_vector).to(torch.long)
    value_vector_torch = torch.tensor(value_vector).to(torch.float32)/100./10000.
    
    return  [index_vector_torch, value_vector_torch, 
             time_vector,longitude_vector, latitude_vector]



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



def test_satellite():
    year = 2019
    month = 8 
    day = 7
    total_day_number  = 2
    time_spacing = 3600/2
    
    data_folder_name = "/Users/yaoyichen/dataset/auto_experiment/experiment_0/satellite/"
    result_folder_name = ""
    output_file_name = f'satellite_data_{str(year)}{str(month).zfill(2)}{str(day).zfill(2)}_{total_day_number}_72_46.npz'
    
    
    [index_vector, value_vector, time_vector,longitude_vector, latitude_vector] = \
        generate_satellite_data(year, month, day, 
                                    total_day_number, time_spacing,
                    data_folder_name, result_folder_name, output_file_name)
    
    
    plot_lonlat_onearth(longitude_vector,latitude_vector,
                        value_vector, dot_size = 20 )
    

def test_obspack():
    
    year = 2019
    month = 8 
    day = 7
    total_day_number  = 5
    time_spacing = 3600/2
    data_folder_name = "/Users/yaoyichen/dataset/carbon/ObsPack/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18/data/daily/"
    
    result_folder_name = ""
    output_file_name = f'satellite_data_{str(year)}{str(month).zfill(2)}{str(day).zfill(2)}_{total_day_number}_72_46.npz'
    
    
    
    [index_vector, value_vector, time_vector,longitude_vector, latitude_vector] = \
        generate_obspack_data(year, month, day, 
                                    total_day_number, time_spacing,
                    data_folder_name, result_folder_name, output_file_name)
    
    
    plot_lonlat_onearth(longitude_vector,latitude_vector,
                        value_vector, dot_size = 20 )
    
    

#%%

def obspack_reader(folder_name, file_list):
    value_list = []
    time_list = []
    longitude_list = []
    latitude_list = []
    
    for file_ in file_list:
    # file_ = file_list[0
        
        file_name = os.path.join(folder_name,
                                 file_)
        df = nc.Dataset(file_name)
        
        
        longitude = np.asarray(df.variables["longitude"][:])
        latitude = np.asarray(df.variables["latitude"][:])
        time_original = np.asarray(df.variables["time"][:])
        time_ = [plus_seconds(int(x)) for x in time_original]
        """
        测量距离地面的高度 
        """
        intake_height = np.asarray(df.variables["intake_height"][:])
        
        valid_index = (intake_height<1000) & (intake_height >1)
        
        model_sample_window_start = np.asarray(df.variables["model_sample_window_start"][:])
        model_sample_window_end =  np.asarray(df.variables["model_sample_window_end"][:])
        start_time = [plus_seconds(int(x)) for x in model_sample_window_start]
        end_time  =  [plus_seconds(int(x)) for x in model_sample_window_end]
        
        value  = np.asarray(df.variables["value"][:])
    
    
        value_list += list(value[valid_index])
        for i in range(len(valid_index)):
            if(valid_index[i] == True):
                # print(time, time[i])
                time_list += [time_[i]]
        longitude_list += list(longitude[valid_index])
        latitude_list += list(latitude[valid_index])
        
    longitude_vector = np.asarray(longitude_list)
    latitude_vector = np.asarray(latitude_list)
    value_vector = np.asarray(value_list)
    return longitude_vector,latitude_vector, value_vector,time_list




def generate_obspack_data(year, month, day, 
                            total_day_number, time_spacing,
                            data_folder_name, result_folder_name, output_file_name):

    
    time_length = int(total_day_number*86400/time_spacing)
    start_datetime =  datetime.datetime(year, month, day, 0,0,0)
    file_list = os.listdir(data_folder_name)
    file_list = clean_file_list(file_list,endsstring = ".nc")
    
    selected_file_list = []
    for step_ in range ( total_day_number):
        step_datetime = start_datetime + datetime.timedelta(days = step_)
        str_datetime = step_datetime.strftime("%Y%m%d")
        print(str_datetime)
        
        if_found = 0
        
        for file_str in file_list:
            if(str_datetime in file_str):
                print(f"finding satellite date:{str_datetime}")
                selected_file_list.append(file_str)
                if_found = 1
                break 
        if(if_found == 0):
            print(f"missing satellite date:{str_datetime}")
    
    
    longitude_vector,latitude_vector, value_vector, time_list = \
        obspack_reader(data_folder_name, selected_file_list)
    
    
    
    longitude_vector,latitude_vector,time_vector = assign2grid( 
        latitude_spacing = 4,
        longitude_spacing = 5,
        time_spacing = time_spacing,
        start_time =  datetime.datetime(year, month ,day, 0,0,0),
        latitude = latitude_vector,
        longitude = longitude_vector,
        time_vector = time_list)
    
    
    value_vector,longitude_vector,latitude_vector, time_vector,result_dict =  \
    average_duplicatd_obs(longitude_vector,latitude_vector,time_vector, value_vector,  
                map_shape = (time_length, 72, 46),max_clip = None, min_clip = None)
    
    
    
    index_list = []
    for time_value, longitude_value, latitude_value in zip(time_vector, longitude_vector, latitude_vector):
        """
        np.ravel_multi_index -> 得到一维向量编码 
        np.unravel_index     -> 得到多维向量编码  
        """
        index_result = np.ravel_multi_index((time_value,longitude_value, latitude_value),  (240,72,46))
        index_list.append(index_result)
        


    index_vector = np.asarray(index_list)
    
    longitude_vector = longitude_vector*5 - 180 
    latitude_vector = latitude_vector*4 - 90
    
    
    index_vector_torch = torch.tensor(index_vector).to(torch.long)
    value_vector_torch = torch.tensor(value_vector).to(torch.float32)
    
    return [index_vector_torch, value_vector_torch, time_vector,longitude_vector, latitude_vector]
    
    
#%%
if __name__ == '__main__':
    test_satellite()
    # test_obspack()
    
    
