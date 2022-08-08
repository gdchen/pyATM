#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:39:11 2022

@author: yaoyichen
"""


#%%
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
from dynamic_model.Transport import Basic_Transport_Model, transport_simulation,ERA5_transport_Model,era5_transport_simulation, ERA5_transport_Model_3D,era5_transport_simulation_3d,era5_transport_simulation_3d_mixing
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d

from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude

from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw

from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco

from file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem, write_carbon_netcdf_2d_avg
from carbon_tools import construct_state_with_cinit,construct_state_with_fluxbottom, construct_state_with_cinitfluxbottom, construct_state_with_cinitfluxbottom_flux1time
from carbon_tools  import get_bottom_flux, get_c_init
from carbon_tools  import construct_Merra2_initial_state_3d 


from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw

from scipy.interpolate import interpn




class Args:
    method = "rk4"
    adjoint = False
    year = 2019
    month = 7
    day = 1
    last_day = 6
    interval_minutes = 20

args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



folder_name = "../data/nc_file/"
# file_name = "myfile_carbon.nc"
preserve_layers = 47

simulation_len = int(args.last_day*1440//args.interval_minutes)


device = "cpu"
merra2_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
geoschem_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
# merra2_folder = "/home/eason.yyc/data/carbon_inversion/201907/merra2/"
# geoschem_folder = "/home/eason.yyc/data/carbon_inversion/201907/concentration/"
# device = "cuda"
def construct_uvw_all_3d(merra2_folder):
    """
    读入气象数据
    """
    # folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    # folder_name = "/home/eason.yyc/data/carbon_inversion/201907/merra2"
    startswith_string = "GEOSChem.StateMet.201907"
    preserve_layers = 47
    uvw_vector =  get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"],preserve_layers= preserve_layers )

    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]

    pbl_top = get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = ["Met_PBLTOPL",])


    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:preserve_layers]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:preserve_layers]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:preserve_layers]
    pbl_top = torch.tensor(pbl_top, dtype= torch.float32)


    return  u_all, v_all, w_all,pbl_top




# 初始化网格信息，以及时间信息

grid_info, time_info = construct_Merra2_initial_state_3d(folder_name = merra2_folder,
                                                         file_name = "GEOSChem.StateMet.20190701_0000z.nc4",
                                                         year = args.year,
                                                         month = args.month,
                                                         day = args.day,
                                                         last_day = args.last_day,
                                                         interval_minutes = args.interval_minutes,
                                                         preserve_layers = preserve_layers)

time_vector, time_string, nt_time = time_info
dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor = grid_info





#%%
"""
# step0: 初始化filed, 读入气象数据
#        初始化 c_init
         初始化 bottom_flux
        
"""


keep_vertical = [0,10,20]
# keep_vertical = [0, 10, 27]

vector_z = vector_z[keep_vertical]
grid_z = grid_z[:,:,keep_vertical]
map_factor = map_factor[:,:,keep_vertical]

#%%
u_all, v_all, w_all,pbl_top = construct_uvw_all_3d(merra2_folder )
# u_all = torch.zeros([432, 72, 46, 47])
# v_all = torch.zeros([432, 72, 46, 47])
# w_all = torch.zeros([432, 72, 46, 47])
# pbl_top = torch.zeros([432, 1, 46, 47])

u_all  = u_all[:,:,:,keep_vertical]
v_all  = v_all[:,:,:,keep_vertical]
w_all = w_all[:,:,:,keep_vertical]
pbl_top = pbl_top[:,:,:,keep_vertical]

# geoschem_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'

geoschem_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"

longitude_vector, latitude_vector = get_variable_Merra2_vector_single(geoschem_folder, geoschem_file,variable_list = ["lon","lat"])


#%%  正向的画图
# 这边注意 c_init 和 bottom_flux的来源
# c_init = get_c_init( source_config = "geos-chem", folder_name=geoschem_folder, file_name = geoschem_file,keep_vertical = keep_vertical)

long_index = 54
lati_index = 16

c_init = 0.000*torch.ones([1, 72, 46, len(keep_vertical)])
bottom_flux = get_bottom_flux(source_config = "init_constant")
bottom_flux[0,long_index,lati_index,0] = 1e-10
state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
variable_list = [state_all,time_vector,dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor]

for variable in variable_list:
    variable = variable.to(device)
    
model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 3)

model = model.to(device)


# 开始数值仿真
with torch.no_grad():
    print(f"start simulation")
    start_time = time.time()
    total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")    

write_netcdf_2d(total_result.detach()[-1,1,:,:,0], f"../data/nc_file/simulation_forward_{long_index}_{lati_index}.nc", longitude_vector, latitude_vector)

#%%


long_index = 6
lati_index = 25

c_init = 0.000*torch.ones([1, 72, 46, len(keep_vertical)])
bottom_flux_first = 0.0*torch.ones([1, 72, 46, 1])
bottom_flux_first.requires_grad = True

bottom_flux_temp = get_bottom_flux(source_config = "init_constant")
bottom_flux = torch.concat( [bottom_flux_first, bottom_flux_temp[1::,:,:,:]] )


state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
variable_list = [state_all,time_vector,dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor]

# for variable in variable_list:
    
state_all = state_all.to(device)
time_vector = time_vector.to(device)
grid_x = grid_x.to(device)
grid_y = grid_y.to(device)
grid_z = grid_z.to(device)
vector_x = vector_x.to(device)
vector_y = vector_y.to(device)
vector_z = vector_z.to(device)
map_factor = map_factor.to(device)


model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 3)

model = model.to(device)


# 开始数值仿真
# optimizer = Adam([bottom_flux_first], lr=0.002)
optimizer = Adam([bottom_flux_first], lr=1.0)
criterion = torch.nn.MSELoss()

print(f"start simulation")
start_time = time.time()

from torch.autograd.functional import jacobian, vjp
def temp_transfer_function(bottom_flux_first):
    bottom_flux_temp = get_bottom_flux(source_config = "init_constant")
    bottom_flux = torch.concat( [bottom_flux_first, bottom_flux_temp[1::,:,:,:]] )
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
    
    total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
    loss = criterion(total_result[47*9,1,long_index,lati_index,0] , torch.tensor([1.0]))
    return loss

result = jacobian(temp_transfer_function, (bottom_flux_first))

write_netcdf_2d(-1.0*result.detach()[0,:,:,0], f"../data/nc_file/simulation_jacobian_{long_index}_{lati_index}.nc", longitude_vector, latitude_vector)





