# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys


sys.path.append("..") 
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
import time
from torch.optim import Adam
from dynamic_model.initialize_tools import gkern
from dynamic_model.Transport import Basic_Transport_Model, transport_simulation,ERA5_transport_Model,era5_transport_simulation, ERA5_transport_Model_3D,era5_transport_simulation_3d
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d

from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude

from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw

from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num

"""
"""

class Args:
    method = "rk4"
    # niters = 1000
    # viz = True
    adjoint = False
    # load_model = False
    # prefix = "ode2d_noadjoint_"
    # checkpoint_file_name = "./checkpoint/ode/" + prefix + "test1.pth.tar"

args = Args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


folder_name = "../data/nc_file/"
file_name = "myfile_carbon.nc"
preserve_layers = 47


#%% 坐标, flux, c, u, v 时间序列

def construct_Merra2_initial_state_3d():
    """
    初始化网格向量，初始化时间轴
    """
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离
    dx = (6378)*1000 * 2.0*np.pi / 360.0
    dy = dx

    folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    variable_list = ["Met_U", "Met_V","Met_OMEGA"] 
    file_name = "GEOSChem.StateMet.20190701_0000z.nc4"


    x, y = get_variable_Merra2_vector_single(folder_name, file_name,variable_list = ["lon","lat"])
    # x, y, u, v, w = get_uvw()

    # print( x.shape, y.shape)

    vector_x = torch.tensor(x * dx, dtype = torch.float32)
    vector_y = torch.tensor(y * dy, dtype = torch.float32)
    dz = 1
    # vector_z = torch.tensor([0.0, 1000,2000,3000,4000,5500,7000,9000, 13000, 20000])[0:preserve_layers]
    vector_z = -1.0*torch.tensor([0.9925,0.9775,	0.9625,	0.9475,	0.9325,	0.9175,	0.9025,	0.8875,	0.8725,
    	0.8575,	0.8425,	0.8275,	0.8100,	0.7875,	0.7625,	0.7375,	0.7125,	0.6875,	0.6563,	0.6188,	0.5813,
        	0.5438,	0.5063,	0.4688,	0.4313,	0.3938,	0.3563,	0.3128,	0.2665,	0.2265,	0.1925,	0.1637,
            	0.1391,	0.1183,	0.1005,	0.0854,	0.0675,	0.0483,	0.0343,	0.0241,	0.0145,	0.0067,	0.0029,
                	0.0011,	0.0004,	0.0001,0.0000])[0:preserve_layers]*100000
    
    print(vector_x, vector_y, vector_z)

    grid_x, grid_y, grid_z = torch.meshgrid(vector_x, vector_y, vector_z)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))


    latitude_cut_factor = 3.0
    map_factor[map_factor > latitude_cut_factor] =   latitude_cut_factor
    map_factor[map_factor < 0.0] =   0.0

    grid_info = (dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor)

    #### time info, 20分钟间隔, 7天  ####
    delta_second = 20*60
    nt_time = int(3*24*6) -1
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1)

    time_string = [datetime.datetime(2019, 7, 1) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]

    time_info = time_vector, time_string, nt_time
    return grid_info, time_info




def construct_cuv_all_3d():
    
    folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    startswith_string = "GEOSChem.StateMet.201907"
    u, v, w =  get_variable_Merra2_3d_batch(folder_name, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = ["Met_U", "Met_V","Met_OMEGA"] )

    preserve_layers = 47
    """
    w方向速度降为很小很小
    """
    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:preserve_layers]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:preserve_layers]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:preserve_layers]


    folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    file_name = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
    variable_list = ["SpeciesConc_CO2"]
    [c] = get_variable_Merra2_3d_single(folder_name, file_name, latitude_dim = 46, longitude_dim = 72, variable_list = variable_list)
    
    c_all = torch.tensor(c, dtype= torch.float32).repeat([len(u_all),1,1,1])[:,:,:,0:preserve_layers]

    f_all = torch.zeros(c_all.shape, dtype= torch.float32)[:,:,:,0:preserve_layers]

    # f_all  = add_flux_m02(f_all)

    return f_all, c_all, u_all, v_all, w_all

# 初始化网格信息，以及时间信息
grid_info, time_info = construct_Merra2_initial_state_3d()
time_vector, time_string, nt_time = time_info
dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor = grid_info


# 初始化filed

# exit
f_all, c_all, u_all, v_all, w_all = construct_cuv_all_3d( )
state_all = torch.stack([f_all, c_all, u_all, v_all, w_all]).permute([1,0,2,3,4])

print(state_all.shape)

model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 3)


# 开始数值仿真
with torch.no_grad():
    print(f"start simulation")
    start_time = time.time()
    total_result = era5_transport_simulation_3d(model, time_vector, state_all)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")

print(total_result.shape)



#%%
def write_carbon_netcdf_3d_geoschem(data_, output_nc_name, time_string, plot_interval, longitude_vector, latitude_vector, vector_z):

    # df = nc.Dataset(ref_nc_name)

    # original_nx, original_ny, original_nt = df.dimensions[
    #     "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
    # new_nx, new_ny = original_nx, original_ny

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





folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare/OutputDir/'
variable_list = ["Met_U", "Met_V"] 
file_name = "GEOSChem.StateMet.20190701_0000z.nc4"


longitude_vector, latitude_vector = get_variable_Merra2_vector_single(folder_name, file_name,variable_list = ["lon","lat"])
    
    
    
write_carbon_netcdf_3d_geoschem(data_=total_result.detach().numpy(),
             output_nc_name='../data/nc_file/upwind_rk2_05.nc',
             time_string=time_string, plot_interval=1,
             longitude_vector = longitude_vector,
             latitude_vector = latitude_vector,
             vector_z = vector_z[0:preserve_layers])


write_carbon_netcdf_2d_geoschem(data_=total_result.detach().numpy(),
             output_nc_name='../data/nc_file/upwind_rk2_2d_05.nc',
             time_string=time_string, plot_interval=1,
             longitude_vector = longitude_vector,
             latitude_vector = latitude_vector)

#%%
# if(True):
#     exit()