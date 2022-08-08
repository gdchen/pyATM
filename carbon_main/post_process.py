#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:40:03 2022

@author: yaoyichen
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import torch
import matplotlib.pyplot as plt





def postprocess(input_nc, output_nc, day_number, frame_number, result_type ):
    """
    几个入参分别是啥意思，看不懂了

    """
    ncin = Dataset(input_nc, 'r', 'NETCDF4')
    f_tensor = ncin["f"][:]
    f_tensor_reshape = f_tensor.reshape([day_number, frame_number, 46, 72, 1])
    
    cin_time = ncin["time"][:]
    
    if(result_type == "day"):
        # 每天一份文件
        f_result = np.mean(f_tensor_reshape, axis = (1))
        time_result = cin_time[0:len(cin_time):frame_number]
        
        
    if(result_type == "frame"):
        # 每个phase一份文件
        f_result = np.mean(f_tensor_reshape, axis = (0))
        time_result = cin_time[0:frame_number]
    
    ncout = Dataset(output_nc, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(ncin["longitude"][:]))
    ncout.createDimension('latitude',  len(ncin["latitude"][:]))
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    timevar.setncattr('units', "minutes since 2019-07-01 00:00:00")
    
    f = ncout.createVariable("f", 'float32',
                                  ('time', 'latitude', 'longitude'))

    f[:] = f_result
    timevar[:] = time_result
    latvar[:] = ncin["latitude"][:]
    lonvar[:] = ncin["longitude"][:]
    
    ncout.close()
    
    return 0


input_nc  = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_inversion/inversion_result_044.nc"
output_nc_day   = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_inversion/aggregate_day.nc"
output_nc_frame = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_inversion/aggregate_frame.nc"
day_number = 5
frame_number = 48


postprocess(input_nc, output_nc_day,   day_number, frame_number, result_type = "day")
postprocess(input_nc, output_nc_frame, day_number, frame_number, result_type = "frame")



