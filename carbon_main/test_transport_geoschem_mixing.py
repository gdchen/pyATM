# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
file_name = "myfile_carbon.nc"
preserve_layers = 47

simulation_len = int(args.last_day*1440//args.interval_minutes)



def construct_uvw_all_3d():
    """
    读入气象数据
    """
    folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    startswith_string = "GEOSChem.StateMet.201907"
    preserve_layers = 47
    uvw_vector =  get_variable_Merra2_3d_batch(folder_name, startswith_string, latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"],preserve_layers= preserve_layers )

    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]

    pbl_top = get_variable_Merra2_3d_batch(folder_name, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = ["Met_PBLTOPL",])


    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:preserve_layers]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:preserve_layers]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:preserve_layers]
    pbl_top = torch.tensor(pbl_top, dtype= torch.float32)


    return  u_all, v_all, w_all,pbl_top




# 初始化网格信息，以及时间信息
merra2_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'

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


keep_vertical = [0, 10, 20, 30, 40]
# keep_vertical = [0, 10, 27]

vector_z = vector_z[keep_vertical]
grid_z = grid_z[:,:,keep_vertical]
map_factor = map_factor[:,:,keep_vertical]

u_all, v_all, w_all,pbl_top = construct_uvw_all_3d( )

u_all  = u_all[:,:,:,keep_vertical]
v_all  = v_all[:,:,:,keep_vertical]
w_all = w_all[:,:,:,keep_vertical]
pbl_top = pbl_top[:,:,:,keep_vertical]

geoschem_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
geoschem_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"

longitude_vector, latitude_vector = get_variable_Merra2_vector_single(geoschem_folder, geoschem_file,variable_list = ["lon","lat"])



# 这边注意 c_init 和 bottom_flux的来源
c_init = get_c_init( source_config = "geos-chem", folder_name=geoschem_folder, file_name = geoschem_file,keep_vertical = keep_vertical)


# bottom_flux = get_bottom_flux(source_config = "init_constant")
bottom_flux_ct_filename = "/Users/yaoyichen/dataset/carbon/carbonTracker/CT2019B.flux1x1.201807_reshape.npy"
bottom_flux = get_bottom_flux(source_config = "carbon_tracker", 
                              file_name = bottom_flux_ct_filename)



#%%
# step1: 验证正向模型
state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 3)

# 开始数值仿真
with torch.no_grad():
    print(f"start simulation")
    start_time = time.time()
    total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")

print(total_result.shape)


save_result = True
if(save_result):
    write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                 output_nc_name='../data/nc_file/simulation_result_3d_0322.nc',
                 time_string=time_string, plot_interval=1,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z[0:preserve_layers])
    
    
    write_carbon_netcdf_2d_geoschem(data_=total_result.detach().numpy(),
                 output_nc_name='../data/nc_file/simulation_result_2d_0322.nc',
                 time_string=time_string, plot_interval=1,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector)

#%% 孪生问题
# 
bottom_flux_one = 0.0*torch.ones([1,72, 46])
bottom_flux_one.requires_grad = True
state_all = construct_state_with_cinitfluxbottom_flux1time(c_init, bottom_flux_one, u_all, v_all,w_all)

# 构造的观测数据
# concentration_tensor = torch.mean(total_result[:, 1:2, :,:,:], dim = (1,4))
# oco_value_vector_torch = concentration_tensor.flatten()[oco_index_vector]  


#%%
from torch.autograd.functional import jacobian, vjp

def temp_transfer_function(bottom_flux_one):
    state_all = construct_state_with_cinitfluxbottom_flux1time(c_init, bottom_flux_one, u_all, v_all,w_all)
    state_full_pred = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    concentration_tensor = torch.mean(state_full_pred[:, 1:2, :,:,:], dim = (1,4))
    concentration_tensor_observe = concentration_tensor.flatten()[oco_index_vector] 
    
    # reduce_mean = torch.mean(concentration_tensor_observe)
    return concentration_tensor_observe

# start_time = time.time()
# result = jacobian(temp_transfer_function, (bottom_flux_one))
# end_time = time.time()

start_time = time.time()
# result = jacobian(temp_transfer_function, (bottom_flux_one))
result = vjp(temp_transfer_function,(bottom_flux_one), oco_value_vector_torch)
end_time = time.time()
print( end_time - start_time)

 
#%%
criterion = torch.nn.MSELoss()
criterion_mae = torch.nn.L1Loss()

for iteration in range(1):
    # 需要重新赋值
    state_all = construct_state_with_cinitfluxbottom_flux1time(c_init, bottom_flux_one, u_all, v_all,w_all)
    
    print(f"iteration:{iteration}")
    start_time = time.time()
    
    state_full_pred = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    
    # 变量选择浓度, 在变量和高度方向平均
    concentration_tensor = torch.mean(state_full_pred[:, 1:2, :,:,:], dim = (1,4))
    concentration_tensor_observe = concentration_tensor.flatten()[oco_index_vector]   
    loss = criterion(concentration_tensor_observe, oco_value_vector_torch)
    print( loss.item() )

#%%

bottom_flux_one_save = bottom_flux_one.grad 
#%% 读入卫星观测数据, 只做同化

# 需要打成 index_vector 和 value_vector
satellite_filename = "/Users/yaoyichen/project_earth/carbon/satellite_data_20190701_432_72_46.npz"
oco_index_vector, oco_value_vector , oco_time_vector = get_oco(satellite_filename)  
oco_value_vector_torch = torch.tensor(oco_value_vector)/10000/100.

#%% 
if(False):
    """
    配置1：
    CO2全场均匀初始化
    flux采用bottom_up flux 
    """
    c_init = get_c_init(source_config = "init_constant", keep_vertical=keep_vertical)
    bottom_flux_ct_filename = "/Users/yaoyichen/dataset/carbon/carbonTracker/CT2019B.flux1x1.201807_reshape.npy"
    bottom_flux = get_bottom_flux(source_config = "carbon_tracker", 
                                  file_name = bottom_flux_ct_filename)


if(False):
    """
    配置2：
    CO2从文件开始读取
    flux采用全场为0初始化
    """
    c_init = get_c_init(source_config = "from_file",
                        folder_name="", file_name = "c_init_0313_11_2026.pt",
                        keep_vertical=keep_vertical)
    bottom_flux = get_bottom_flux(source_config = "init_constant")
    

#%% 同时优化 c_init  和 bottom_flux

inversion_type = "only_flux"

if(inversion_type == "only_flux"):
    bottom_flux.requires_grad = True
    c_init.requires_grad = False
    optimizer_fluxbottom = Adam([bottom_flux], lr=2.0e-11)

if(inversion_type == "only_initfield"):
    bottom_flux.requires_grad = False
    c_init.requires_grad = True
    optimizer_cinit = Adam([c_init], lr=1.0e-5)

if(inversion_type == "flux&initfield"):
    bottom_flux.requires_grad = True
    c_init.requires_grad = True
    optimizer_fluxbottom = Adam([bottom_flux], lr=2.0e-11)
    optimizer_cinit = Adam([c_init], lr=1.0e-5)



#%%
inversion_type = None
criterion = torch.nn.MSELoss()

for iteration in range(500):
    # 需要重新赋值
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
    
    print(f"iteration:{iteration}")
    start_time = time.time()
    
    state_full_pred = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top, if_mixing=False)
    
    # 变量选择浓度, 在变量和高度方向平均
    concentration_tensor = torch.mean(state_full_pred[:, 1:2, :,:,:], dim = (1,4))
    concentration_tensor_observe = concentration_tensor.flatten()[oco_index_vector]   
    loss = criterion(concentration_tensor_observe, oco_value_vector_torch)
    print( loss.item() )
    
    if(inversion_type == "only_flux"):
        optimizer_fluxbottom.zero_grad()
        loss.backward()
        optimizer_fluxbottom.step()
        
    
    if(inversion_type == "only_initfield"):
        optimizer_cinit.zero_grad()
        loss.backward()
        optimizer_cinit.step()
        
        
    if(inversion_type == "flux&initfield"):
        optimizer_fluxbottom.zero_grad()
        optimizer_cinit.zero_grad()
        
        loss.backward()
        
        optimizer_fluxbottom.step()
        optimizer_cinit.step()


    print(f"elapse time:{time.time() - start_time: .3e}")
    
    # write_carbon_netcdf_2d_geoschem(data_=state_full_pred.detach().numpy(),
    #               output_nc_name=f'../data/nc_file/quick_see_result_2d_flux_layer5_{str(iteration+2000).zfill(4)}.nc',
    #               time_string=time_string, plot_interval=18,
    #               longitude_vector = longitude_vector,
    #               latitude_vector = latitude_vector)
    
    torch.save(c_init,f"../data/field_file/flux_layer5_{str(iteration+2000).zfill(4)}.pt")



#%%
write_carbon_netcdf_2d_geoschem(data_=state_full_pred.detach().numpy(),
              output_nc_name='../data/nc_file/quick_see_result_2d.nc',
              time_string=time_string, plot_interval=18,
              longitude_vector = longitude_vector,
              latitude_vector = latitude_vector)


#%%

write_carbon_netcdf_2d_avg(data_= state_full_pred.detach(),
             output_nc_name='../data/nc_file/quick_see_result_2d_avg_2.nc',
             time_string=time_string, plot_interval=1,
             longitude_vector = longitude_vector,
             latitude_vector = latitude_vector,
             average_interval = 432)



#%% 
"""
这段是计算gap的，没什么用
"""
gap_tensor = torch.abs(oco_value_vector_torch- concentration_tensor_observe.detach())
time_divide = 72
result = {}

for time,gap in zip(oco_time_vector, gap_tensor):
    day_index = int(time.item()//time_divide)
    if(day_index not in result):
        result[day_index] = [gap.item()]
    else:
        result[day_index].append(gap.item())
        
for day, value in result.items():
    print(day,np.mean(value))
    
    
#%%   
torch.save(c_init,"c_init_0313_11_layer5.pt")
torch.save(bottom_flux,"bottom_flux_0312_23.pt")
    