#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:16:28 2022

@author: yaoyichen
"""

# 读入carbon tracker 数据， 并能够差值到现有的网格上

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import torch
import cv2
import matplotlib.pyplot as plt
import os

folder_name = "/Users/yaoyichen/dataset/carbon/carbonTracker/total_co2/"
file_list = ["CT2019B.molefrac_glb3x2_2018-01-01.nc",
             "CT2019B.molefrac_glb3x2_2018-02-01.nc",
             "CT2019B.molefrac_glb3x2_2018-03-01.nc",
             "CT2019B.molefrac_glb3x2_2018-04-01.nc",
             "CT2019B.molefrac_glb3x2_2018-05-01.nc",
             "CT2019B.molefrac_glb3x2_2018-06-01.nc",
             "CT2019B.molefrac_glb3x2_2018-07-01.nc",
             "CT2019B.molefrac_glb3x2_2018-08-01.nc",
             "CT2019B.molefrac_glb3x2_2018-09-01.nc",
             "CT2019B.molefrac_glb3x2_2018-10-01.nc",
             "CT2019B.molefrac_glb3x2_2018-11-01.nc",
             "CT2019B.molefrac_glb3x2_2018-12-01.nc"]

co2_mean_merge = np.zeros([25,12])
co2_std_merge  = np.zeros([25,12])
pressure_merge = np.zeros([26,12])

for index, file_ in enumerate(file_list):
    input_file = os.path.join(folder_name, file_)
    df = nc.Dataset(input_file)
    
    co2 = df["co2"][:]
    pressure =  df["pressure"][:]
    
    co2_mean = co2.mean(axis = (0,2,3))
    co2_std  = co2.std(axis = (0,2,3))
    pressure_mean = pressure.mean(axis = (0,2,3))
    co2_mean_merge[:, index] = co2_mean
    co2_std_merge[:, index] = co2_std
    pressure_merge[:, index] = pressure_mean
    
    plt.figure(0)
    plt.plot(co2_mean, label = index)
    plt.figure(1)
    plt.plot(co2_std, label = index)
    plt.legend()
    
co2_mean_merge_mean = np.mean(co2_mean_merge, axis = 1)
co2_std_merge_mean  = np.mean(co2_std_merge, axis = 1)
pressure_merge_mean = np.mean(pressure_merge, axis = 1)

pressure_profile = pressure_merge_mean/pressure_merge_mean.max()


pressure_merra2 = np.asarray([0.9925,0.9775,	0.9625,	0.9475,	0.9325,	0.9175,	0.9025,	0.8875,	0.8725,
	0.8575,	0.8425,	0.8275,	0.8100,	0.7875,	0.7625,	0.7375,	0.7125,	0.6875,	0.6563,	0.6188,	0.5813,
    	0.5438,	0.5063,	0.4688,	0.4313,	0.3938,	0.3563,	0.3128,	0.2665,	0.2265,	0.1925,	0.1637,
        	0.1391,	0.1183,	0.1005,	0.0854,	0.0675,	0.0483,	0.0343,	0.0241,	0.0145,	0.0067,	0.0029,
            	0.0011,	0.0004,	0.0001,0.0000])

# interpolate_2d(x_grid_1d_pad, y_grid_1d_pad, u_use_pad_yx, x_point, y_point)

from scipy.interpolate import interpn     

points_ref = (pressure_profile[::-1],)

  
merra2_mean_vector = interpn(points_ref,
                  np.concatenate([co2_mean_merge_mean, np.asarray([co2_mean_merge_mean[-1]])])[::-1],
                                                    pressure_merra2, method = "linear")
merra2_mean_vector = merra2_mean_vector - np.mean(merra2_mean_vector)
merra2_mean_vector = merra2_mean_vector/1000000.

merra2_std_vector =  interpn(points_ref, 
                  np.concatenate([co2_std_merge_mean ,np.asarray([co2_std_merge_mean[-1]])])[::-1], 
                                                    pressure_merra2, method = "linear")
merra2_std_vector = merra2_std_vector/1000000.

plt.figure(2)
plt.plot(merra2_mean_vector, label = index)
plt.figure(3)
plt.plot(merra2_std_vector, label = index)


with open('/Users/yaoyichen/dataset/auto_experiment/experiment_0/carbontracker/ct_mean_std_profile.npy', 'wb') as f:
    np.save(f, merra2_mean_vector)
    np.save(f, merra2_std_vector)
    


#%% 读数据的模块
with open('/Users/yaoyichen/dataset/auto_experiment/experiment_0/carbontracker/ct_mean_std_profile.npy', 'rb') as f:
    merra2_mean_vector = np.load(f)
    merra2_std_vector  = np.load(f)

#%% 以下为算协方差的模块

# co2_layer = np.transpose(co2[0,0,:,:], (1,0))
# co2_layer_flu = co2_layer - np.mean(co2_layer)
# plt.imshow( np.flip(co2_layer_flu, axis = (0)))

# H_size = 5
# L_size = 3
# matrix = np.zeros([H_size*2+1, L_size*2 +1 ])
# for i in range(-H_size,H_size+1,1):
#     for j in range(-L_size, L_size+1 ,1):
#         co2_layer_flu_shift_x = np.roll(co2_layer_flu, i - H_size , axis=0)
#         co2_layer_flu_shift_xy = np.roll(co2_layer_flu_shift_x, j - L_size , axis=1)
#         result = co2_layer_flu_shift_xy*co2_layer_flu/np.sqrt(co2_layer_flu**2*co2_layer_flu_shift_xy**2)
#         matrix[i,j] = np.mean(result)
        

# plt.imshow(matrix)
        
        

