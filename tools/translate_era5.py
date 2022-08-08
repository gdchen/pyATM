#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:53:24 2021

@author: yaoyichen
"""

#%%
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num

### 缩放倍数
span_x ,span_y,span_t = 1, 1, 1
variable_list = ["u","v","z","t","cc"]


### 
file_name2 = "../nc_file/adaptor.mars.internal-1633759268.6305075-18218-10-884b2b4d-0868-4812-b8a6-74736ed8fac5.nc"
df = nc.Dataset(file_name2)

for variable in df.variables:
    print(variable)
    print(variable, df.variables[variable].long_name)

#%%
original_nx, original_ny,original_nt  = df.dimensions["longitude"].size , df.dimensions["latitude"].size, df.dimensions["time"].size
print( original_nx, original_ny,original_nt)

# 取2天
original_nt = int(original_nt*2/7) + 1

### process
new_lonvar = df.variables["longitude"][:][0:original_nx:span_x]
new_latvar = df.variables["latitude"][:][0:original_ny:span_y][::-1]
new_timevar = df.variables["time"][:][0:original_nt:span_t]
print(new_lonvar, new_latvar)
new_nx, new_ny = len(new_lonvar), len(new_latvar)
print( f"new nx:{new_nx}, new ny:{new_ny}")

## write
ncout = Dataset('myfile_carbon_3d.nc','w','NETCDF4'); # using netCDF3 for output format 
ncout.createDimension('longitude',new_nx)
ncout.createDimension('latitude',new_ny)
ncout.createDimension('time',None)


lonvar = ncout.createVariable('longitude','float32',('longitude'))
latvar = ncout.createVariable('latitude','float32',('latitude'))
timevar = ncout.createVariable('time','int32',('time'))


lonvar.setncattr('units',df.variables["longitude"].units)
latvar.setncattr('units',df.variables["latitude"].units)
timevar.setncattr('units',df.variables["time"].units)

lonvar[:] = new_lonvar
latvar[:] = new_latvar
timevar[:]= new_timevar


for variable_name in variable_list:
    tt = ncout.createVariable(variable_name,'float32',('time','latitude','longitude'))
    tt.setncattr('units',df.variables[variable_name].units)
    values = df.variables[variable_name][:][0:original_nt:span_t,0:original_ny:span_y, 0:original_nx:span_x]
    tt[:] = values[:,::-1]
    

ncout.close()
