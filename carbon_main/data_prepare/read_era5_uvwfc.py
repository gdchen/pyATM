#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:11:52 2022

@author: yaoyichen
"""
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_variable_carbon_3d_uvw(folder_name, file_name):
    """
    原有的顺序 时间，level，latitude, longtitude
    修改后的顺序  时间，longtitude，latitude, level

    """
    # file_name = "../data/nc_file/myfile_carbon.nc"
    folder_file_name = os.path.join(folder_name, file_name)

    df = nc.Dataset(folder_file_name)
    longitude = df.variables["longitude"][:]
    latitude = df.variables["latitude"][:]
    level = df.variables["level"][:]

    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    level = np.asarray(df.variables["level"][:])
    print(f"longitude:{longitude}")
    print(f"latitude:{latitude}")
    print(f"level:{latitude}")

    u = np.transpose(np.asarray(df.variables["u"][:]), (0, 3, 2, 1))
    v = np.transpose(np.asarray(df.variables["v"][:]), (0, 3, 2, 1))
    w = np.transpose(np.asarray(df.variables["w"][:]), (0, 3, 2, 1))

    print(u.shape)
    

    return longitude, latitude, u, v, w


def field_intepolation(variable, factor = 2):
    """
    factor 时间方向做差值放大，放大倍数
    """
    variable_shape= list(variable.shape)
    
    variable_shape[0] = (variable_shape[0]-1)*factor +1 
    
    result = np.zeros(variable_shape)
    for i in range(len(variable) -1):
        for j in range(factor):
            next_ratio = 1.0*j/factor
            previous_ratio = 1.0 - next_ratio
            result_temp = previous_ratio* variable[i,::] + next_ratio * variable[i+1,::]
            # print(factor*i + j, i,i+1, previous_ratio, next_ratio)
            result[factor*i + j,::] = result_temp
    result[-1,::] = variable[-1,::]
    return result



#%%

def get_variable_carbon_3d_bottomup_flux(folder_name, file_name,span_x = 1 ,span_y = 1):
    """
    latitude -90 --> 90 间隔 1°
    longitude 0 --> 360 间隔 1°
    span_x = 1 为采样间隔， 将 原始 1°的结果采样的更粗 
    span_y = 1 为采样间隔， 将 原始 1°的结果采样的更粗 
    """
    
    folder_file_name = os.path.join(folder_name, file_name)
    
    df = nc.Dataset(folder_file_name)
    latitude_len = df.dimensions["latitude"].size
    longitude_len = df.dimensions["longitude"].size
    
    print(df.variables["latitude"][:], df.variables["longitude"][:])
    flux = np.transpose(np.asarray(df.variables["emi_co2"][:]), ( 1, 0))[0:longitude_len:span_x, 0:latitude_len:span_y  ]
    return flux





#%%
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num



def reorder_geoschem_co2_data(folder_name, file_name,out_nc_name, level_list):
    """
    将 geos-chem中自带的文件重新排序 
    原始 
    latitude -90 --> 90 间隔 2°  91
    longitude -180 --> 180  间隔2.5°   144 
    
    重新排序后 
    latitude -90 --> 90 间隔 2°  91
    longitude 0 -> 360 间隔2.5°   144 
    
    高度方向按照 level_list 来抽取

    """
    folder_file_name = os.path.join(folder_name, file_name)
    
    df = nc.Dataset(folder_file_name)
    latitude_len = df.dimensions["lat"].size
    longitude_len = df.dimensions["lon"].size
    
    
    original_nx, original_ny = df.dimensions[
            "lon"].size, df.dimensions["lat"].size
    
    
    new_nx, new_ny = original_nx, original_ny
    levels = len(level_list)
    ncout = Dataset(os.path.join(folder_name, out_nc_name), 'w', 'NETCDF4')
    
    ncout.createDimension('longitude', new_nx)
    ncout.createDimension('latitude', new_ny)
    ncout.createDimension('level', levels)
    
    
    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    levelvar = ncout.createVariable('level', 'float32', ('level'))
    
    
    lonvar.setncattr('units', df.variables["lon"].units)
    latvar.setncattr('units', df.variables["lat"].units)
    levelvar.setncattr('units', df.variables["ilev"].units)
    
    # 经度将负号改成正号
    lon_temp = df.variables["lon"][:]
    lon_modify = np.concatenate([lon_temp[len(lon_temp)//2: len(lon_temp)] ,360+ lon_temp[0:len(lon_temp)//2]] )
    lonvar[:] = lon_modify
    
    # 纬度方向不做调整
    latvar[:] = df.variables["lat"][:]
      
    
    # 高度方向选取期望的高度方向                
    level_origine = df.variables["ilev"][:]
    levelvar[:] = level_origine.take(level_list, axis = 0)
    
    
    c = ncout.createVariable("c", 'float32',
                                 ('level','latitude', 'longitude')) 
    
    c_origine =  df["SpeciesRst_CO2"][:]
    c_temp = c_origine[0].take(level_list, axis = 0)
    
    c_modify = np.concatenate([c_temp[:,:,len(lon_temp)//2: len(lon_temp)] ,c_temp[:,:, 0:len(lon_temp)//2]] , axis = 2)
    c[:] = c_modify
    
    ncout.close()
    return 0



#%% 将原有结果做空间差值到指定形态


"""
将数据切换到仿真代码可用的格式
"""
def get_variable_carbon_3d_concentration(folder_name, file_name, latitude_dim = 91, longitude_dim = 180):
    """
    原始的顺序维 ('level','latitude', 'longitude'), 
    更改后的顺序维 ('longitude'，'latitude'， 'level')

    中间有差值模块
    """
    folder_file_name = os.path.join(folder_name, file_name)
    df = nc.Dataset(folder_file_name)
    c = df["c"][:]
    c_shape = list(c.shape)
    c_shape[2] = longitude_dim
    c_shape[1] = latitude_dim
    result = np.zeros(c_shape)
    
    print("#"*20, c.shape)
    
    original_level, original_latitude_len, original_longitude_len = c.shape
    if((original_latitude_len != latitude_dim) or (original_longitude_len != longitude_dim) ):
        print("need to reshape")
        for i in range(len(c)):
            result[i,:,:] = cv2.resize(c[i,:,:], dsize=(longitude_dim, latitude_dim), interpolation=cv2.INTER_CUBIC)
    print("#"*20, result.shape)
    result_reorder = np.transpose(result ,( 2,1,0))
    return result_reorder



def get_variable_Merra2_3d_single(folder_name, file_name, latitude_dim = 91, longitude_dim = 180, variable_list = ["Met_U", "Met_V"]):
    """
    读入单个 Merra2文件
    原始的顺序维 ('level','latitude', 'longitude'), 
    更改后的顺序维 ('longitude'，'latitude'， 'level')

    从Merra2数据中，读入 uv 数据
    """
    folder_file_name = os.path.join(folder_name, file_name)
    # print(folder_file_name)
    df = nc.Dataset(folder_file_name)
    
    final_result = []
    for variable in variable_list:
        # print(variable_list, variable)
        vector_u = df[variable][:]
        if(vector_u.shape[0] == 1):
            vector_u = np.squeeze(vector_u, axis=0)
        
        c_shape = list(vector_u.shape)

        """
        如果是三维数据
        """
        if(len(c_shape) == 3):
            c_shape[2] = longitude_dim
            c_shape[1] = latitude_dim
        elif(len(c_shape) == 2):
            c_shape[1] = longitude_dim
            c_shape[0] = latitude_dim
        
        result_u = np.zeros(c_shape)
        
        if(len(c_shape) == 3):
            original_level, original_latitude_len, original_longitude_len = list(vector_u.shape)
        elif(len(c_shape) == 2):
            original_latitude_len, original_longitude_len = list(vector_u.shape)

        if((original_latitude_len != latitude_dim) or (original_longitude_len != longitude_dim) ):
            print(f"need to reshape, original shape:{original_longitude_len, original_latitude_len}, \
                  target shape :{longitude_dim, latitude_dim} ")
            if(len(c_shape) == 3):
                for i in range(len(vector_u)):
                    result_u[i,:,:] = cv2.resize(vector_u[i,:,:], dsize=(longitude_dim, latitude_dim), interpolation=cv2.INTER_CUBIC)
            elif(len(c_shape) == 2):
                result_u[:,:] = cv2.resize(vector_u[:,:], dsize=(longitude_dim, latitude_dim), interpolation=cv2.INTER_CUBIC)

        else:
            result_u = vector_u

        if(len(c_shape) == 3):
            result_u = np.transpose(result_u ,( 2,1,0))
        elif(len(c_shape) == 2):
            result_u = np.transpose(result_u ,( 1,0))

        final_result.append(result_u)
    return final_result



def get_variable_Merra2_3d_batch(folder_name, startswith_string,latitude_dim = 46, longitude_dim = 72,
                                 variable_list = ["Met_U", "Met_V"],args = None,
                                preserve_layers = None ):   
    # file_name_list_coarse = os.listdir(folder_name)
    # file_name_list = []
    # for file_name in file_name_list_coarse:
    #     if(file_name.startswith(startswith_string)):
    #         file_name_list.append(file_name)
    start_datetime =  datetime.datetime(args.year, args.month, args.day,0,0,0)
    total_step_number = int(args.last_day*1440/args.interval_minutes)
    file_name_list = []
    # print(args.year, args.month, args.day,total_step_number)
    for step_ in range ( total_step_number):
        step_datetime = start_datetime + datetime.timedelta(minutes = 30*step_)
        str_datetime = step_datetime.strftime("%Y%m%d_%H%M")
        filename_ = startswith_string + str_datetime + "z.nc4"
        # print(filename_)
        file_name_list.append(filename_)
            
    result = []
    # 按名字排序了
    file_name_list.sort()
    
    result_u_list = []    
    for file_name in file_name_list:
        result_u = get_variable_Merra2_3d_single(folder_name, file_name,latitude_dim, longitude_dim,variable_list)
        result_u_list.append(result_u)
    result = np.asarray(result_u_list)
    return result, file_name_list


def get_variable_Merra2_vector_single(folder_name, file_name, variable_list = ["lon", "lat"]):
    """
    读入任何数组, 可以是1维, 2维或者3维
    
    """
    final_result = []
    folder_file_name = os.path.join(folder_name, file_name)
    df = nc.Dataset(folder_file_name)
    
    for varible in variable_list:
        varible_np = np.asarray(df.variables[varible][:])
        final_result.append(varible_np)
    return np.asarray(final_result)




def reorder_geoschem():
    folder_name = '/Users/yaoyichen/dataset/GeosChem'
    file_name = "GEOSChem.Restart.20190701_0000z.nc4"
    out_nc_name = "GEOSChem.Restart.20190701_0000z_reorder.nc4"
    level_list = [0,7,13,17,20,23,25,28,32,38]

    reorder_geoschem_co2_data(folder_name, file_name, out_nc_name, level_list)


def get_c():
    folder_name = '/Users/yaoyichen/dataset/GeosChem'
    file_name = "GEOSChem.Restart.20190701_0000z_reorder.nc4"
    result = get_variable_carbon_3d_concentration(folder_name, file_name,latitude_dim = 91, longitude_dim = 180 )
    print(f"concentration shape:{result.shape}")
    return result +  3.395352377988522e-05


def get_c_point(f_in):
    result = torch.zeros(f_in.shape[1::])

    for x in range(120,140):
        for y in range(55,75):
            if((x - 130)**2 + (y - 65)**2 < 5**2):
                result[ x,y,0] = result[ x,y,0] + 4.0e-5
    return result


def get_c_zero(f_in):
    result = torch.zeros(f_in.shape[1::])
    return result


def get_uvw():
    folder_name = '/Users/yaoyichen/dataset/era5/'
    file_name = "era5_3d_2022_firstweek_8x.nc"

    # 读入变量，并在时间方向按需要差值
    longitude, latitude, u_orginal, v_orginal, w_orginal = get_variable_carbon_3d_uvw(folder_name, file_name)
    u = field_intepolation(u_orginal, factor = 1)
    v = field_intepolation(v_orginal, factor = 1)
    w = field_intepolation(w_orginal, factor = 1)

    print(f"u shape {u.shape}" )
    return longitude, latitude, u, v, w


def get_Merra2_uvw():
    folder_name = '/Users/yaoyichen/dataset/'
    file_name = "era5_3d_2022_firstweek_8x.nc"

    # 读入变量，并在时间方向按需要差值
    longitude, latitude, u_orginal, v_orginal, w_orginal = get_variable_carbon_3d_uvw(folder_name, file_name)
    u = field_intepolation(u_orginal, factor = 1)
    v = field_intepolation(v_orginal, factor = 1)
    w = field_intepolation(w_orginal, factor = 1)

    print(f"u shape {u.shape}" )
    return longitude, latitude, u, v, w



def get_f():
    span_x ,span_y = 2, 2
    folder_name = "/Users/yaoyichen/project_earth/carbon_bottomup/results/redistribute"
    file_name = "redistribute_2020_09_result.nc"

    result = get_variable_carbon_3d_bottomup_flux(folder_name, file_name, span_x = span_x ,span_y = span_y)
    print(f"flux shape:{result.shape}")
    return result



def get_oco(filename):
    # file_name = "/Users/yaoyichen/project_earth/carbon/oco_data_2022_first_week.npz"
    data = np.load(filename)
    index_vector = torch.tensor(data['index_vector'], dtype = torch.long)
    value_vector = torch.tensor(data['value_vector'], dtype = torch.float32)
    time_vector = torch.tensor(data['time_vector'], dtype = torch.float32)
    return index_vector, value_vector,time_vector


# result_f = get_f()
# c = get_c()
# get_uvw()
# index_vector, value_vector = get_oco()
# print(a,b)
# print(c.mean())
