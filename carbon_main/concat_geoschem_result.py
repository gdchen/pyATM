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
将所有geos-chem 输出结果串联起来的小程序
"""


# nc_name_list = [f"result_simulation_result_3d_0210_full_{i}.nc" for i in range(300)]

def concat_geoschem_result(folder_name, file_name_list, variable_list,output_nc_name):
    
    
    ref_nc_name = os.path.join(ref_folder, nc_name_list[0])

    df = nc.Dataset(ref_nc_name)
    
    original_nx, original_ny, original_nt,original_level = df.dimensions[
        "lon"].size, df.dimensions["lat"].size, df.dimensions["time"].size, df.dimensions["lev"].size
    new_nx, new_ny = original_nx, original_ny
    
    
    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('lon', new_nx)
    ncout.createDimension('lat', new_ny)
    ncout.createDimension('lev', original_level)
    ncout.createDimension('time', None)
    
    
    lonvar = ncout.createVariable('lon', 'float32', ('lon'))
    latvar = ncout.createVariable('lat', 'float32', ('lat'))
    levvar = ncout.createVariable('lev', 'float32', ('lev'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    
    lonvar.setncattr('units', df.variables["lon"].units)
    latvar.setncattr('units', df.variables["lat"].units)
    levvar.setncattr('units', df.variables["lev"].units)
    timevar.setncattr('units', df.variables["time"].units)
    
    
    lonvar[:] = df.variables["lon"][:]
    latvar[:] = df.variables["lat"][:]
    levvar[:] = df.variables["lev"][:]
    
    timevar[:] = np.arange(15, 15 + 30*336,30)
    # timevar[:] = np.arange(180,180 + 360*124,360)
    
    for variable in  variable_list:
        f = ncout.createVariable(variable, 'float32',
                                 ('time', 'lev','lat', 'lon'))
        
        for index, value in enumerate(nc_name_list):
            
            ref_nc_name = os.path.join(ref_folder, value)
            df = nc.Dataset(ref_nc_name)
            f[index,:,:,:] = df[variable][:,:,:,:]
            
    
    
    ncout.close()
    
    
    
def concat_geoschem_result_2d(folder_name, file_name_list, variable_list,output_nc_name,need_mean = False):
    
    
    ref_nc_name = os.path.join(ref_folder, nc_name_list[0])

    df = nc.Dataset(ref_nc_name)
    
    original_nx, original_ny, original_nt,original_level = df.dimensions[
        "lon"].size, df.dimensions["lat"].size, df.dimensions["time"].size, df.dimensions["lev"].size
    new_nx, new_ny = original_nx, original_ny
    
    
    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('lon', new_nx)
    ncout.createDimension('lat', new_ny)
    ncout.createDimension('time', None)
    
    
    lonvar = ncout.createVariable('lon', 'float32', ('lon'))
    latvar = ncout.createVariable('lat', 'float32', ('lat'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    
    lonvar.setncattr('units', df.variables["lon"].units)
    latvar.setncattr('units', df.variables["lat"].units)
    timevar.setncattr('units', df.variables["time"].units)
    
    
    lonvar[:] = df.variables["lon"][:]
    latvar[:] = df.variables["lat"][:]
    
    # timevar[:] = np.arange(10,10 + 20*505,20)
    timevar[:] = np.arange(15, 15 + 30*336,30)
    # timevar[:] = np.arange(180,180 + 360*124,360)
    
    for variable in  variable_list:
        f = ncout.createVariable(variable, 'float32',
                                 ('time','lat', 'lon'))
        
        for index, value in enumerate(nc_name_list):
            
            ref_nc_name = os.path.join(ref_folder, value)
            df = nc.Dataset(ref_nc_name)
            if(need_mean):
                f[index,:,:] = np.mean(df[variable][:,:,:,:],axis =1)
            else:
                f[index,:,:] = df[variable][:,:,:]
            
    ncout.close()



ref_folder = "/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_addchem/OutputDir/"
# startswith_string = "GEOSChem.StateMet.201907"
# variable_list = ["Met_U","Met_V"]
startswith_string = "GEOSChem.SpeciesConc.201907"
variable_list = ["SpeciesConc_CO2"]


file_name_list_coarse = os.listdir(ref_folder)
nc_name_list = []
for file_name in file_name_list_coarse:
    if(file_name.startswith(startswith_string)):
        nc_name_list.append(file_name)
nc_name_list.sort()            


output_nc_name = "/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_addchem/OutputDir/co2_fossil.nc"
output_nc_name_2d = "/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_addchem/OutputDir/co2_fossil_2d.nc"

# output_nc_name_pbl = "/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/met_result_nomixing_2d_pbl.nc"
                         

concat_geoschem_result(ref_folder, nc_name_list, variable_list, output_nc_name)
concat_geoschem_result_2d(ref_folder, nc_name_list, variable_list, output_nc_name_2d, need_mean= True)

# concat_geoschem_result_2d(ref_folder, nc_name_list, variable_list=["Met_PBLTOPL"], 
# output_nc_name = output_nc_name_pbl, need_mean= False)