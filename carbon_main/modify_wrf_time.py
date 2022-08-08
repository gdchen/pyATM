#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:21:40 2022

@author: yaoyichen
"""
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import torch
import numpy as np



ncin = Dataset("/Users/yaoyichen/project_earth/data_assimilation/data/nc_file/quick_see_result_2d_flux_layer5_2106.nc", 'r', 'NETCDF4')


ncout = Dataset("/Users/yaoyichen/project_earth/data_assimilation/data/nc_file/tt2.nc", 'w', 'NETCDF4')
ncout.createDimension('longitude', len(ncin["longitude"][:]))
ncout.createDimension('latitude', len(ncin["latitude"][:]))
ncout.createDimension('time', None)

lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
timevar = ncout.createVariable('time', 'int32', ('time'))


timevar.setncattr('units', "minutes since 2019-07-01 00:00:00")


lonvar[:] = ncin["longitude"][:]
latvar[:] = ncin["latitude"][:]

calendar = 'standard'
timevar[:] = ncin["time"][:][0]

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


f[:] = np.mean(ncin["f"][0:14,:,:], axis = 0)
c[:] = np.mean(ncin["c"][0:14,:,:], axis = 0)
u[:] = np.mean(ncin["u"][0:14,:,:], axis = 0)
v[:] = np.mean(ncin["v"][0:14,:,:],axis = 0) 
w[:] = np.mean(ncin["w"][0:14,:,:], axis = 0)

ncout.close()

