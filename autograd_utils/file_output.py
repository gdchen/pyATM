#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:25:36 2022

@author: yaoyichen
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import torch

def write_carbon_netcdf_3d_geoschem(data_, output_nc_name, time_string, plot_interval, longitude_vector, latitude_vector, vector_z):


    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(longitude_vector))
    ncout.createDimension('latitude', len(latitude_vector))
    ncout.createDimension('height', len(vector_z))
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    heightvar = ncout.createVariable('height', 'float32', ('height'))
    timevar = ncout.createVariable('time', 'int32', ('time'))

    # lonvar.setncattr('units', df.variables["longitude"].units)
    # latvar.setncattr('units', df.variables["latitude"].units)
    # heightvar.setncattr('units', "")
    timevar.setncattr('units', "minutes since 2019-07-01 00:00:00")

    total_len = len(time_string)

    lonvar[:] = longitude_vector
    latvar[:] = latitude_vector
    heightvar[:] = vector_z

    calendar = 'standard'
    timevar[:] = nc.date2num(time_string[
        0:total_len:plot_interval], units = "minutes since 2019-07-01 00:00:00",calendar=calendar)


    f = ncout.createVariable("f", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    c = ncout.createVariable("c", 'float32',
                             ('time', 'latitude', 'longitude','height')) 
    u = ncout.createVariable("u", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    v = ncout.createVariable("v", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    w = ncout.createVariable("w", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    

    f[:] = np.transpose(data_[0:total_len:plot_interval, 0, :, :, :], (0, 2, 1,3))
    c[:] = np.transpose(data_[0:total_len:plot_interval, 1, :, :, :], (0, 2, 1,3))
    u[:] = np.transpose(data_[0:total_len:plot_interval, 2, :, :, :], (0, 2, 1,3))
    v[:] = np.transpose(data_[0:total_len:plot_interval, 3, :, :, :], (0, 2, 1,3))
    w[:] = np.transpose(data_[0:total_len:plot_interval, 4, :, :, :], (0, 2, 1,3))

    ncout.close()
    return None







def write_carbon_netcdf_2d_geoschem(data_, output_nc_name, time_string, plot_interval, longitude_vector, latitude_vector):

    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(longitude_vector))
    ncout.createDimension('latitude', len(latitude_vector))
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))


    timevar.setncattr('units', "minutes since 2019-07-01 00:00:00")

    total_len = len(time_string)

    lonvar[:] = longitude_vector
    latvar[:] = latitude_vector

    calendar = 'standard'
    timevar[:] = nc.date2num(time_string[
        0:total_len:plot_interval], units = "minutes since 2019-07-01 00:00:00",calendar=calendar)


    f = ncout.createVariable("f", 'float32',
                             ('time', 'latitude', 'longitude'))
    c = ncout.createVariable("c", 'float32',
                             ('time', 'latitude', 'longitude')) 
    u = ncout.createVariable("u", 'float32',
                             ('time', 'latitude', 'longitude'))
    v = ncout.createVariable("v", 'float32',
                             ('time', 'latitude', 'longitude'))
    w = ncout.createVariable("w", 'float32',
                             ('time', 'latitude', 'longitude'))
    

    f[:] = np.mean(np.transpose(data_[0:total_len:plot_interval, 0, :, :, :], (0, 2, 1,3)), axis = 3)
    c[:] = np.mean(np.transpose(data_[0:total_len:plot_interval, 1, :, :, :], (0, 2, 1,3)), axis = 3)
    u[:] = np.mean(np.transpose(data_[0:total_len:plot_interval, 2, :, :, :], (0, 2, 1,3)), axis = 3)
    v[:] = np.mean(np.transpose(data_[0:total_len:plot_interval, 3, :, :, :], (0, 2, 1,3)), axis = 3)
    w[:] = np.mean(np.transpose(data_[0:total_len:plot_interval, 4, :, :, :], (0, 2, 1,3)), axis = 3)

    ncout.close()
    return None




#%%  输出2维的平均通量

def write_carbon_netcdf_2d_avg(data_,
             output_nc_name,
             time_string, 
             plot_interval,
             longitude_vector,
             latitude_vector ,
             average_interval):

    data_len = len(data_)
    
    result = torch.empty((1,data_.shape[1],data_.shape[2],data_.shape[3],1))
    
    for i in np.arange(0,data_len,average_interval):
        print(i,i+average_interval )
        tt = data_[i:(i+average_interval),:,:,:,:]
        # print("aaa",tt.shape)
        aa = torch.mean(data_[i:(i+average_interval),:,:,:,:],dim = (0,4)).unsqueeze(0).unsqueeze(-1)
        result = torch.cat([result,aa])
    
    time_string_temp = time_string[0:len(time_string):average_interval]
    
    write_carbon_netcdf_2d_geoschem(result.numpy(),
                 output_nc_name,
                 time_string=time_string_temp, plot_interval=1,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector)
    return 0
