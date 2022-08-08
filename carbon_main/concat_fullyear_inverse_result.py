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


ref_folder_list = \
["/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180101/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180116/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180201/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180220/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180301/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180316/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180401/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180416/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180501/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180516/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180601/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180616/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180701/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180716/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180801/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180816/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180901/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180916/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181001/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181016/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181101/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181116/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181201/",
"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20181216/",
]

# /Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_20180116

nc_name_list = []
for ref_folder in ref_folder_list:
    ref_nc_name = os.path.join(ref_folder, "inversion_result_069.nc")
    nc_name_list.append(ref_nc_name)

# /Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc

output_nc_name = "/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/full2018_inversion.nc"

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
timevar.setncattr('units', "minutes since 2018-01-01 00:00:00")
# lonvar.setncattr('units', df.variables["longitude"].units)
# latvar.setncattr('units', df.variables["latitude"].units)
# timevar.setncattr('units', df.variables["time"].units)

lonvar[:] = df.variables["longitude"][:]
latvar[:] = df.variables["latitude"][:]


# timevar[:] = np.arange(len(nc_name_list))

f = ncout.createVariable("f", 'float32',
                         ('time', 'latitude', 'longitude'))

time_var_list = []
for index, value in enumerate(nc_name_list):
    
    
    ref_nc_name = os.path.join(value)
    df = nc.Dataset(ref_nc_name)
    f[index,:,:] = df['f'][0,:,:]
    # time_var_list.append(df["time"][0])

calendar = 'standard'

time_string = [datetime.datetime(2018, 1,1),
                datetime.datetime(2018, 1,16),
                datetime.datetime(2018, 2,1),
                datetime.datetime(2018, 2,20),
                datetime.datetime(2018, 3,1),
                datetime.datetime(2018, 3,16),
                datetime.datetime(2018, 4,1),
                datetime.datetime(2018, 4,16),
                datetime.datetime(2018, 5,1),
                datetime.datetime(2018, 5,16),
                datetime.datetime(2018, 6,1),
                datetime.datetime(2018, 6,16),
                datetime.datetime(2018, 7,1),
                datetime.datetime(2018, 7,16),
                datetime.datetime(2018, 8,1),
                datetime.datetime(2018, 8,16),
                datetime.datetime(2018, 9,1),
                datetime.datetime(2018, 9,16),
                datetime.datetime(2018, 10,1),
                datetime.datetime(2018, 10,16),
                datetime.datetime(2018, 11,1),
                datetime.datetime(2018, 11,16),
                datetime.datetime(2018, 12,1),
                datetime.datetime(2018, 12,16)]

                   
timevar[:] = nc.date2num(time_string, units = "minutes since 2018-01-01 00:00:00",calendar=calendar)
# timevar[:] = np.arange(time_var_list)


ncout.close()