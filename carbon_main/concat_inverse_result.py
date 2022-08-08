#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:49:53 2022

@author: yaoyichen
"""

import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
import datetime
import numpy as np
import os


"""
将所有反演结果串联起来的小程序
"""

ref_folder = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_satellite/"

nc_name_list = [f"inversion_result_x{str(i).zfill(3)}.nc" for i in range(0,300,1)]
print(nc_name_list)

ref_nc_name = os.path.join(ref_folder, nc_name_list[0])
output_nc_name = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_satellite/result.nc"

print(f"ref_nc_name:{ref_nc_name}")
df = nc.Dataset(ref_nc_name)

original_nx, original_ny, original_nt = df.dimensions[
    "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
new_nx, new_ny = original_nx, original_ny


ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
ncout.createDimension('longitude', new_nx)
ncout.createDimension('latitude', new_ny)
ncout.createDimension('time', None)


lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
timevar = ncout.createVariable('time', 'int32', ('time'))

# lonvar.setncattr('units', df.variables["longitude"].units)
# latvar.setncattr('units', df.variables["latitude"].units)
# timevar.setncattr('units', df.variables["time"].units)

lonvar[:] = df.variables["longitude"][:]
latvar[:] = df.variables["latitude"][:]


timevar[:] = np.arange(len(nc_name_list))

f = ncout.createVariable("f", 'float32',
                         ('time', 'latitude', 'longitude'))

for index, value in enumerate(nc_name_list):
    
    ref_nc_name = os.path.join(ref_folder, value)
    df = nc.Dataset(ref_nc_name)
    f[index,:,:] = df['f'][0,:,:]


ncout.close()