#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:53:35 2022

@author: yaoyichen
"""

import numpy as np



def modify_oco_latitude(latitude_spacing, latitude, latitude_map):
    result = []
    for value in latitude:
        # print(f"value: {value}")
        nearest_value = -90 + latitude_spacing*int(((value+90) + 0.5*latitude_spacing)//latitude_spacing)
        # print(f"nearest_value:{nearest_value}")
        index= latitude_map[nearest_value]
        result.append(index)
        # print(f"index:{index}")
    return np.asarray(result)


def modify_oco_longitude(longitude_spacing, longitude, longitude_map):
    result = []
    for value in longitude:
        # if(value < 0):
        #     value += 360
        # print(f"value: {value}")
        nearest_value = -180 + (longitude_spacing*int(((value+180) + 0.5*longitude_spacing)//longitude_spacing))%360
        # print(f"nearest_value:{nearest_value}")
        index = longitude_map[nearest_value]
        # print(f"index:{index}")
        # index= -180 + longitude_map[longitude_spacing*int(((value+180) + 0.5*longitude_spacing)//longitude_spacing)]
        result.append(index)
    return np.asarray(result)


def modify_oco_time(time_vector, start_time,time_spacing):
    result = []
    for time_value in time_vector:
        time_delta = time_value - start_time
        hour_value = int(time_delta.total_seconds()//time_spacing)
        result.append(hour_value)

    return np.asarray(result)



def assign2grid(latitude_spacing, longitude_spacing,time_spacing, start_time, latitude,longitude,time_vector):
    """
    latitude_spacing   °
    longitude_spacing  °
    longitude_spacing  秒
    
    结果的经纬度排列  latitude  -90 到  90
                    longitude  -180 到 180
    
    """
    latitude_grids = 180//latitude_spacing + 1
    longitude_grids = 360 // longitude_spacing
    
    
    latitude_map = {}
    for i in range(latitude_grids):
        latitude_map[latitude_spacing*i - 90] = i
    
    longitude_map = {}
    for i in range(longitude_grids):
        longitude_map[longitude_spacing*i -180] = i
        
    # print(latitude_map)
    latitude_vector = modify_oco_latitude(latitude_spacing,latitude, latitude_map)
    longitude_vector = modify_oco_longitude(longitude_spacing, longitude, longitude_map)
    time_vector = modify_oco_time(time_vector, start_time,time_spacing)
    
    return longitude_vector,latitude_vector,time_vector




def averaging_obs(longitude_vector, latitude_vector, hour_vector,value_vector):
    """
    数据去重，如果重复则设置为平均值
    """
    result_dict = {}
    for longitude, latitude, hour,value in zip(longitude_vector, latitude_vector, hour_vector,value_vector):
        key = "_".join([str(longitude), str(latitude), str(hour)])
        if(not key in result_dict):
            result_dict[key] = {"value":value, "count":1}
        else:
            original_count= result_dict[key]["count"]
            original_value= result_dict[key]["value"]
            
            result_count = original_count+ 1
            result_value = 1.0*(original_count*original_value + value)/result_count
            result_dict[key] = {"value":result_value, "count":result_count}
    return result_dict


def unsplit_average_str(result_dict,  map_shape = (7*24*3,180,91), max_clip = None, min_clip = None):
    index_list, value_list,longitude_list,latitude_list,time_list = [],[],[],[],[]
    for key,value in result_dict.items():
        temp = key.split("_")
        
        longitude_value, latitude_value, time_value  = int(temp[0]),int(temp[1]),int(temp[2])
        
        # index_result = np.ravel_multi_index((int(temp[2]),int(temp[0]), int(temp[1])), map_shape)
        # index_list.append(index_result)
        
        value = value['value']
        if(max_clip):
            value = min(max_clip,value)
        if(min_clip):
            value = max(min_clip,value)
            
        value_list.append(value)
        longitude_list.append(longitude_value)
        latitude_list.append(latitude_value)
        time_list.append(time_value)
        
    
    # index_vector = np.asarray(index_list)
    value_vector = np.asarray(value_list)
    longitude_vector =  np.asarray(longitude_list)
    latitude_vector =  np.asarray(latitude_list)
    time_vector =  np.asarray(time_list)
    
    return value_vector,longitude_vector,latitude_vector, time_vector


def average_duplicatd_obs(longitude_vector, latitude_vector, hour_vector,value_vector, map_shape = (168,180,91), max_clip = None, min_clip = None):
    result_dict = averaging_obs(longitude_vector, latitude_vector, hour_vector,value_vector)
    # print(result_dict)
    value_vector,longitude_vector,latitude_vector, time_vector = unsplit_average_str(result_dict, map_shape, max_clip, min_clip)
    return value_vector,longitude_vector,latitude_vector, time_vector, result_dict


if __name__ == "__main__":
    pass
    # latitude_spacing = 4
    # longitude_spacing = 5
    # start_time = datetime.datetime(2022,1,1,0,0,0)
    # assign2grid(latitude_spacing,longitude_spacing,)
    
