#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:41:45 2022

@author: yaoyichen
"""

import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import cv2
import torch
import matplotlib.pyplot as plt
from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem
from autograd_utils.geoschem_tools  import calculate_x_result, calculate_weight_z
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 


def get_variable_Merra2_3d_single(folder_name, file_name, 
                                  latitude_dim = 91, longitude_dim = 180, 
                                  variable_list = ["Met_U", "Met_V"], time_index = 0):
    """
    读入单个 Merra2文件
    原始的顺序维 ('level','latitude', 'longitude'), 
    更改后的顺序维 ('longitude'，'latitude'， 'level')

    从Merra2数据中，读入 uv 数据
    """
    folder_file_name = os.path.join(folder_name, file_name)
    df = nc.Dataset(folder_file_name)
    
    final_result = []
    for variable in variable_list:
        # print(variable_list, variable)
        vector_u = df[variable][:]
        if(vector_u.shape[0] == 1):
            vector_u = np.squeeze(vector_u, axis=0)
            
        vector_u = vector_u[time_index,::].squeeze()
        
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

result_2d_merge = []
for index in range(60):
    [tt] = get_variable_Merra2_3d_single("/Users/yaoyichen/dataset/comparison",
                                  "GEOSChem.SpeciesConc.20190701_0020z.nc4",
                                  variable_list = ["SpeciesConc_CO2"],time_index= index
                                  )
    tt2 = np.mean(tt, axis = 2)
    result_2d_merge.append(tt2)


data_folder= '/Users/yaoyichen/dataset/auto_experiment/experiment_0/'
merra2_folder = os.path.join(data_folder, 'merra2')
file_name = "GEOSChem.StateMet.20190701_0000z.nc4"
grid_info, time_info = construct_Merra2_initial_state_3d(folder_name = merra2_folder,
                                                         file_name = file_name,
                                                         year = 2019,
                                                         month = 7,
                                                         day = 1,
                                                         last_day = 30,
                                                         interval_minutes = 30)

time_vector, time_string, nt_time = time_info
(longitude_vector,latitude_vector,
             dx, dy, dz, _1, _2, _3, 
             vector_x, vector_y, vector_z_origine, map_factor_origine) = grid_info


from autograd_utils.geoschem_tools import construct_uvw_all_3d,generate_vertical_info,regenerate_vertcal_state

weight_z = calculate_weight_z(vector_z_origine)   

#%%

result = np.asarray(result_2d_merge)

result2 = np.expand_dims(result,(1,4))
result3 = np.concatenate([result2,result2,result2,result2,result2], axis = 1)

x_result = calculate_x_result(torch.tensor(result3), weight_z)

longitude_vector = np.arange(-180,180,2)
latitude_vector = np.arange(-91,91,2)

write_carbon_netcdf_3d_geoschem(data_=x_result.detach().numpy(),
              output_nc_name= os.path.join("gd_2d.nc"),
              time_string=time_string[0:len(time_string):24],
              plot_interval= 1,
              longitude_vector = longitude_vector,
              latitude_vector = latitude_vector,
              vector_z = np.asarray([1.0]))

plt.plot(result3.mean(axis = (1,2,3,4)))


