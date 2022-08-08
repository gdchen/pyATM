#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:30:22 2022

@author: yaoyichen
"""

import os
import sys
import numpy as np
import torch
from torch.autograd.functional import jacobian
from torch.autograd import grad
import torch.nn as nn
from torch.optim import LBFGS, Adam, SGD
from torch.distributions.multivariate_normal import MultivariateNormal
import logging
import time
import matplotlib.pyplot as plt
from dynamic_model.integral_module import RK4Method, JRK4Method, VJRK4Method, da_odeint, rk4_step, Jrk4_step, RK4_Method, da_odeint_boundary, euler_step
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss, classical3dvar
from tools.plot_tools import plot_2d
from dynamic_model.ERA5_v2 import ERA5_pressure, Weatherbench_QG, construct_weatherbench_initial_state
from dynamic_model.ERA5_v2 import write_netcdf, filter_latitude, filter_longitude,SP_Filter

grid_info,  time_info = construct_weatherbench_initial_state()
time_vector, time_string, nt_time = time_info
dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor = grid_info


pde_model = Weatherbench_QG(grid_info)
# pde_model = ERA5_pressure(grid_info)

device = "cpu"


# total_result = torch.empty(tuple(time_vector.shape,) + state0_true.shape)
# total_result[0, :] = state0_true
# state = state0_true


# def simulation(time_vector, init_state):
#     total_result = torch.empty(tuple(time_vector.shape,) + init_state.shape)
#     total_result[0, :] = init_state

#     state = init_state
#     for index, time in enumerate(time_vector[1::]):
#         print(index, state.mean(dim=(1, 2)))
#         dt = time_vector[index + 1] - time_vector[index]
#         state = state + \
#             rk4_step(pde_model, time, dt, state)
        
        

#         # boundary condition of velocity and height
#         state[0, :, 0] = 0.0
#         state[0, :, -1] = 0.0

#         state[1, :, 0] = 0.0
#         state[1, :, -1] = 0.0

#         state[2, :, 0] = torch.mean(state[2, :, 1], dim=(0))
#         state[2, :, -1] = torch.mean(state[2, :, -2], dim=(0))

#         total_result[index + 1, :] = state

#     return total_result


# print(time_vector)
# total_result = simulation(time_vector, state0_true)

# write_netcdf(data_=total_result.detach().numpy(),
#              ref_nc_name= "/Users/yaoyichen/dataset/era5/old/myfile19.nc",
#              output_nc_name='./data/nc_file/result_simulation_final_0406.nc',
#              time_string=time_string, plot_interval=12)

# %%


from torch.utils.data import Dataset, DataLoader
import xarray as xr



import os 
import shutil
from os import walk

def get_filedetail_from_folder(folder_name, endswith_str = None, startswith_str = None):
    """
    获取 folder_name 中所有的有效的文件名,按endswith_str 和 startswith_str 过滤
    
    """

    folder_file_list = [] 
    for (dirpath, dirnames, filenames) in walk(folder_name):
        # if(len(dirnames) == 0):
        #     print(dirpath, dirnames, filenames)
        
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


class QG_Dataset(Dataset):
    """
    Data generator for WeatherBench data.
    Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Args:
        ds: Dataset containing all variables
        var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
        lead_time: Lead time in hours
        batch_size: Batch size
        shuffle: bool. If True, data is shuffled.
        load: bool. If True, datadet is loaded into RAM.
        mean: If None, compute mean from data.
        std: If None, compute standard deviation from data.
        
        不同变量，不同高度，单独做归一化的
    """
    def __init__(self,  var_dict, 
                 data_folder,
                 lead_time = 24, 
                 inner_batch_size = 128, 
                 year_lower = 1979,
                 year_upper = 2018,
                 ):
        # self.ds = ds
        self.var_dict = var_dict
        self.data_folder = data_folder
        self.lead_time = lead_time
        self.inner_batch_size = inner_batch_size
        
        self.file_list = get_filedetail_from_folder(os.path.join(self.data_folder, "geopotential/"),
                                                    endswith_str=".nc")
        self.year_list = []
        for file_name in self.file_list:
            year_int= int( file_name.split("_")[1] ) 
            if((year_int >= year_lower ) and (year_int <= year_upper )):
                self.year_list.append(year_int)
        self.year_list.sort()
        self.year_mapping = {}
        for index, year in enumerate(self.year_list):
            self.year_mapping[index] = year
            
    
        # self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        # self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        # self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # # Normalize
        # if(if_norm ==  True):
        #     self.data = (self.data - self.mean) / self.std
        # self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        # self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        # self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        # 总长度为年份大小 
        # z500 = xr.open_mfdataset(f'{self.data_folder}geopotential/geopotential_2018_5.625deg.nc' , combine='by_coords')
        # self.length = z500.z.shape[0] 
        return len(self.year_list)
        
    def __getitem__(self, index):
        
        year_string = str(self.year_mapping[index])
        z = xr.open_mfdataset(f'{self.data_folder}geopotential/geopotential_{year_string}_5.625deg.nc' , combine='by_coords')
        t = xr.open_mfdataset(f'{self.data_folder}temperature/temperature_{year_string}_5.625deg.nc' , combine='by_coords')
        u = xr.open_mfdataset(f'{self.data_folder}u_component_of_wind/u_component_of_wind_{year_string}_5.625deg.nc' , combine='by_coords')
        v = -xr.open_mfdataset(f'{self.data_folder}v_component_of_wind/v_component_of_wind_{year_string}_5.625deg.nc' , combine='by_coords')

        #t850 需要额外增加 .drop('level')
        datasets = [z, t, u, v]
        ds = xr.merge(datasets)

        # 求平均
        # ds_train = ds.sel(time=slice('2018',"2019"))  
        # ds = ds_train
        
        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in self.var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))
            except KeyError:
                data.append(ds[var])
        
        data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        
        time_index = np.random.choice(np.arange(8760 - self.lead_time), self.inner_batch_size)
        X = np.asarray(data[time_index,:,:,:])
        y = np.asarray(data[time_index + self.lead_time,:,:,:])
        # print(X.shape, y.shape)
        return X, y
    
var_dict = {"u":[500], "v":[500],'z': [500], }

batch_size = 1
num_workers = 0
data_folder = "/Users/yaoyichen/dataset/weatherbench/"
train_dataset = QG_Dataset(var_dict = var_dict, 
                             data_folder= data_folder,
                             lead_time =24*1,
                             inner_batch_size = 1,
                             year_lower= 2018,
                             year_upper = 2018)

train_loader = DataLoader(
    dataset= train_dataset, batch_size= batch_size,
    num_workers= num_workers , 
    pin_memory = False,shuffle= True)


for X, y in train_loader:
    """
    torch.Size([2, 32, 64, 4])
    """
    X = X.squeeze()
    y = y.squeeze()
    X = np.transpose(X, [2,1,0])
    y = np.transpose(y, [2,1,0])
    
    
    
#%%
state0_true = X
state0_true = state0_true.to(torch.float64)
total_result = torch.empty(tuple(time_vector.shape,) + state0_true.shape)
total_result[0, :] = state0_true
state = state0_true

filter  = SP_Filter(dtype = "float64")
def simulation(time_vector, init_state):
    total_result = torch.empty(tuple(time_vector.shape,) + init_state.shape)
    total_result[0, :] = init_state

    state = init_state
    for index, time_value in enumerate(time_vector[1::]):
        print(index, state.mean(dim=(1, 2)))
        
        dt = time_vector[index + 1] - time_vector[index]
        state = state + \
            rk4_step(pde_model, time_value, dt, state)
            
        
        # state[0:1,::]  = 0.9*state[0:1,::] \
        #   + 0.1*filter.forward(state[0:1,::])
        # state[1:2,::]  = 0.9*state[1:2,::] \
        #   + 0.1*filter.forward(state[1:2,::])
        # state[2:3,::]  = 0.9*state[2:3,::] \
        #     + 0.1*filter.forward(state[2:3,::])
        # filter
        # if index < 2:
        #     percentage = 10
        # else:
        # #     percentage = 4
        # state = filter_longitude(state, nx=64, ny=31, percentage=4,dtype = "float64")
        # state = filter_latitude(state,  nx=64, ny=31, percentage=4,dtype = "float64")
    
        #     for index, time in enumerate(time_vector[1::]):
        #         dt = time_vector[index + 1] - time_vector[index]
        #         """
        #         f, u, v, w 从已知信息中读入
        #         """
        #         state[0:1,::] = state_all[index,0:1,::]
        #         state[2:5,::] = state_all[index,2:5,::]

        #         state = state + \
        #             rk2_step(model, time, dt, state)
        
        
        state = state.to(torch.float64)

        # boundary condition of velocity and height
        state[0, :, 0] = 0.0
        state[0, :, -1] = 0.0

        state[1, :, 0] = 0.0
        state[1, :, -1] = 0.0
        
        # state[0, :, 0] = torch.mean(state[0, :, 1], dim=(0))
        # state[0, :, -1] = torch.mean(state[0, :, -2], dim=(0))
        
        # state[1, :, 0] = torch.mean(state[1, :, 1], dim=(0))
        # state[1, :, -1] = torch.mean(state[1, :, -2], dim=(0))
        
        

        state[2, :, 0]  = torch.mean(state[2, :, 1], dim=(0))
        state[2, :, -1] = torch.mean(state[2, :,-2], dim=(0))

        total_result[index + 1, :] = state

    return total_result


print(time_vector)
total_result = simulation(time_vector, state0_true)


#%%

def calculate_score(y_pred, y_true ):

    lat_array = np.asarray([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625,
                -53.4375, -47.8125, -42.1875, -36.5625, -30.9375, -25.3125,
                -19.6875, -14.0625,  -8.4375,  -2.8125,   2.8125,   8.4375,
                14.0625,  19.6875,  25.3125,  30.9375,  36.5625,  42.1875,
                47.8125,  53.4375,  59.0625,  64.6875,  70.3125,  75.9375,
                81.5625,  87.1875])
        
    weights_lat = np.cos(np.deg2rad(lat_array))
    weights_lat = weights_lat/weights_lat.mean()
    square_error = (y_pred - y_true)**2
    weight_square_error = np.einsum("cjw,w->cjw",square_error, weights_lat) 
    result = np.sqrt(np.mean(weight_square_error , axis = (1,2)) )
    return result


result = calculate_score(total_result[24*1,:,:,:], y)
print(result)


#%%
plt.figure(0)
plt.imshow(X[0,:,:])
plt.figure(3)
plt.imshow(y[0,:,:])
plt.figure(10)
plt.imshow(total_result[24*1, 0,:,:])
# for i in range(12):
#     plt.figure(i*10)
#     plt.imshow(total_result[2*i*1, 0,:,:])


# plt.figure(1)
# plt.imshow(X[1,:,:])
# plt.figure(4)
# plt.imshow(y[1,:,:])
# plt.figure(9)
# plt.imshow(total_result[1*24*1, 1,:,:])



# plt.figure(5)
# plt.imshow(X[2,:,:])
# plt.figure(6)
# plt.imshow(y[2,:,:])
# plt.figure(2)
# plt.imshow(total_result[1*24*1, 2,:,:])

#%%

