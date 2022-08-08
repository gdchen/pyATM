#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 19:02:53 2022

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
from dynamic_model.Transport import transport_simulation,ERA5_transport_Model, ERA5_transport_Model_3D
from dynamic_model.Transport import era5_transport_simulation_3d_mixing
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.netcdf_helper import write_netcdf_single_variable_single_time_2d, write_netcdf_single_variable_multi_time_2d

from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d

from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude

from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw

from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco

from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem, write_carbon_netcdf_2d_avg
from autograd_utils.carbon_tools import construct_state_with_cinit,construct_state_with_fluxbottom, construct_state_with_cinitfluxbottom, construct_state_with_cinitfluxbottom_flux1time
from autograd_utils.carbon_tools  import get_bottom_flux, get_c_init
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 
from autograd_utils.carbon_tools  import generate_vertical_vertical_mapping_all, fulfill_vertical_mapping


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


simulation_len = int(args.last_day*1440//args.interval_minutes)


device = "cpu"
merra2_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
geoschem_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
geoschem_co2_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
startswith_string = "GEOSChem.StateMet.201907"
geoschem_statemet_file = "GEOSChem.StateMet.20190701_0000z.nc4"
if_mixing = False
# if_vertical = False
# device = "cuda"
# merra2_folder = "/home/eason.yyc/data/carbon_inversion/201907/merra2/"
# geoschem_folder = "/home/eason.yyc/data/carbon_inversion/201907/concentration/"


vertical_mapping = {"forward":{0:0, 1:2, 2:4, 3:6, 4:8, 5:10,6:20,7:30,8:40}}
# vertical_mapping = {"forward":{0:0}}
# vertical_mapping = generate_vertical_vertical_mapping_all()
vertical_mapping = fulfill_vertical_mapping(vertical_mapping)
preserve_layers = len(vertical_mapping["forward"])
vertical_indexs = vertical_mapping["vertical_list"] 


def construct_uvw_all_3d(merra2_folder, startswith_string):
    """
    读入气象数据， 生成u,v,w场， 以及边界层pbl信息
    """
    inner_preserve_layers = 47
    uvw_vector =  get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"],
                                               preserve_layers = inner_preserve_layers )

    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]

    pbl_top = get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = ["Met_PBLTOPL",])


    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:inner_preserve_layers]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:inner_preserve_layers]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:inner_preserve_layers]

        
    pbl_top = torch.tensor(pbl_top, dtype= torch.float32)


    return  u_all, v_all, w_all,pbl_top




# 初始化网格信息，以及时间信息

grid_info, time_info = construct_Merra2_initial_state_3d(folder_name = merra2_folder,
                                                         file_name = geoschem_statemet_file,
                                                         year = args.year,
                                                         month = args.month,
                                                         day = args.day,
                                                         last_day = args.last_day,
                                                         interval_minutes = args.interval_minutes,
                                                         preserve_layers = 47)

time_vector, time_string, nt_time = time_info
dx, dy, dz, grid_x, grid_y, grid_z_orgine, vector_x, vector_y, vector_z_orgine, map_factor_orgine = grid_info



u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine = construct_uvw_all_3d( merra2_folder, startswith_string)
#%%
"""
# step2: 初始化filed, 读入气象数据
#        初始化 c_init
         初始化 bottom_flux
"""
vector_z = vector_z_orgine[vertical_indexs]
grid_z = grid_z_orgine[:,:,vertical_indexs]
map_factor = map_factor_orgine[:,:,vertical_indexs]


u_all  = u_all_orgine[:,:,:,vertical_indexs]
v_all  = v_all_orgine[:,:,:,vertical_indexs]
w_all  = w_all_orgine[:,:,:,vertical_indexs]



# 需要有个函数，将pbl_top_current 映射到 vertical_mapping 有的index上
pbl_top_clone = pbl_top_orgine.clone()
for key,value in vertical_mapping["forward"].items():
    pbl_top_clone[pbl_top_orgine>= value] = value
pbl_top_clone = pbl_top_clone.to(torch.int)            
         



longitude_vector, latitude_vector = get_variable_Merra2_vector_single(geoschem_folder, geoschem_statemet_file, variable_list = ["lon","lat"])

# 这边注意 c_init 和 bottom_flux的来源
c_init = get_c_init( source_config = "geos-chem", folder_name=geoschem_folder, file_name = geoschem_co2_file, keep_vertical = vertical_indexs  )
bottom_flux = get_bottom_flux(source_config = "init_constant")

#%%

state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

# 讲变量转到device
variable_list = [state_all,time_vector, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor]
for variable in variable_list:
    variable = variable.to(device)

# 开始数值仿真
#%%

model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 2)

model = model.to(device)


with torch.no_grad():
    print(f"start simulation")
    start_time = time.time()
    start_time = time.time()
    total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top_clone, 
                                                       vertical_mapping, if_mixing= if_mixing)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
    
print(total_result.shape)

#%%
save_result = True
if(save_result):
    write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                 output_nc_name='../data/nc_file/simulation_result_3d_benchmark.nc',
                 time_string=time_string, plot_interval=72,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z[0:preserve_layers])

#%%

def euler_forward_matrix(long_index, lati_index, model,u_all, v_all,w_all , pbl_top_,
                         longitude_vector, latitude_vector, vertical_mapping, file_name,if_mixing):
    
    c_init = 0.000*torch.ones([1,u_all.shape[1],u_all.shape[2], u_all.shape[3] ])
    bottom_flux = get_bottom_flux(source_config = "init_constant")
    bottom_flux[0,long_index,lati_index,0] = 1e-10
    
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
    state_all = state_all.to(device)
    
    # 开始数值仿真
    with torch.no_grad():
        total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top_clone, 
                                                           vertical_mapping, if_mixing=if_mixing)
 
    write_netcdf_single_variable_single_time_2d(total_result.detach()[-1,1,:,:,0], file_name , longitude_vector, latitude_vector)
    
    write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                 output_nc_name= file_name[0:-3] + "_3d" + ".nc",
                 time_string=time_string, plot_interval=72,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z[0:preserve_layers])
    return 0

long_index = 13
lati_index = 28
file_name  = f"../data/nc_file/simulation_forward_{long_index}_{lati_index}.nc"
if_mixing = False
euler_forward_matrix(long_index, lati_index, model,u_all, v_all,w_all , pbl_top_clone,
                         longitude_vector, latitude_vector, vertical_mapping, file_name,if_mixing)


#%%


def euler_jacobian_matrix(long_index, lati_index, model,u_all, v_all,w_all , pbl_top_,
                         longitude_vector, latitude_vector, vertical_mapping, file_name,if_mixing):
    """
    计算jacobian矩阵， 
    """

    c_init = 0.000*torch.ones([1, u_all.shape[1],u_all.shape[2], u_all.shape[3] ])
    bottom_flux_first = 0.0*torch.ones([1, u_all.shape[1],u_all.shape[2], 1])
    bottom_flux_first.requires_grad = True
    
    bottom_flux_temp = get_bottom_flux(source_config = "init_constant")
    bottom_flux = torch.concat( [bottom_flux_first, bottom_flux_temp[1::,:,:,:]] )
    
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
    variable_list = [state_all,time_vector,dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor]
    

    optimizer = Adam([bottom_flux_first], lr=1.0)
    criterion = torch.nn.MSELoss()
    
    start_time = time.time()
    
    
    """
    注意下时间的赋值 47*9，  主要目的是为了和 lagragian的结果相契合
    """
    from torch.autograd.functional import jacobian, vjp
    def temp_transfer_function(bottom_flux_first):
        print(f"start simulation")
        bottom_flux_temp = get_bottom_flux(source_config = "init_constant")
        bottom_flux = torch.concat( [bottom_flux_first, bottom_flux_temp[1::,:,:,:]] )
        state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
        
        total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top_clone, 
                                                           vertical_mapping, if_mixing)
        print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
        loss = criterion(total_result[47*9,1,long_index,lati_index,0] , torch.tensor([1.0]))
        return loss
    
    result = jacobian(temp_transfer_function, (bottom_flux_first))
    write_netcdf_single_variable_single_time_2d(-1.0*result.detach()[0,:,:,0], file_name, longitude_vector, latitude_vector)

    return 0

# long_index = 6
# lati_index = 25
# file_name = f"../data/nc_file/simulation_jacobian_{long_index}_{lati_index}.nc"
# euler_jacobian_matrix(long_index, lati_index, model,u_all, v_all,w_all , pbl_top_clone,
#                          longitude_vector, latitude_vector, vertical_mapping, file_name,if_mixing)





