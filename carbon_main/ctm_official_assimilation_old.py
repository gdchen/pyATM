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
from dynamic_model.Transport import transport_simulation,ERA5_transport_Model, ERA5_transport_Model_3D
from dynamic_model.Transport import era5_transport_simulation_3d_mixing
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d

from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude

from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw

from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco

from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem, write_carbon_netcdf_2d_avg
from autograd_utils.carbon_tools import construct_state_with_cinit,construct_state_with_fluxbottom, construct_state_with_cinitfluxbottom, construct_state_with_cinitfluxbottom_fluxalltime
from autograd_utils.carbon_tools  import get_bottom_flux, get_c_init
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 
from autograd_utils.carbon_tools  import generate_vertical_vertical_mapping_all, fulfill_vertical_mapping

class Args:
    adjoint = False
    year = 2019
    month = 7
    day = 1
    last_day = 5
    interval_minutes = 30

args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


simulation_len = int(args.last_day*1440//args.interval_minutes)


device = "cpu"

# experiment_folder= '/Users/yaoyichen/Desktop/auto_experiment/experiment_1/'
experiment_folder= '/home/eason.yyc/data/auto_experiment/experiment_1/'
merra2_folder = os.path.join(experiment_folder, 'merra2')
geoschem_folder = os.path.join(experiment_folder,  'geoschem')
geoschem_co2_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
startswith_string = "GEOSChem.StateMet.201907"
geoschem_statemet_file = "GEOSChem.StateMet.20190701_0000z.nc4"

if_mixing=True
sim_dimension = 3

# device = "cuda"
# merra2_folder = "/home/eason.yyc/data/carbon_inversion/201907/merra2/"
# geoschem_folder = "/home/eason.yyc/data/carbon_inversion/201907/concentration/"

# 9层的数值模拟
# vertical_mapping = {"forward":{0:0, 1:2, 2:4, 3:6, 4:8, 5:10,6:20,7:30,8:40}}
# vertical_mapping = {"forward":{0:0}}
# vertical_mapping = generate_vertical_vertical_mapping_all()

# vertical_mapping = {"forward":{0:0, 1:2, 2:4, 3:6, 4:9, 5:13,6:18,7:25,8:40}}
vertical_mapping = {"forward":{0:0, 1:2, 2:5, 3:10, 4:17, 5:40}}
vertical_mapping = fulfill_vertical_mapping(vertical_mapping)
preserve_layers = len(vertical_mapping["forward"])
vertical_indexs = vertical_mapping["vertical_list"] 


def construct_uvw_all_3d(merra2_folder, startswith_string):
    """
    读入气象数据， 生成u,v,w场， 以及边界层pbl信息
    """
    # inner_preserve_layers = 47
    uvw_vector =  get_variable_Merra2_3d_batch(merra2_folder, startswith_string, 
                                               latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"] )

    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]

    pbl_top = get_variable_Merra2_3d_batch(merra2_folder, startswith_string, 
                                           latitude_dim = 46, longitude_dim = 72,
                                           variable_list = ["Met_PBLTOPL",])

    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:47]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:47]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:47]
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

#%%
print("loading merra2 data...")
u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine \
    = construct_uvw_all_3d( merra2_folder, startswith_string )
#%%
"""
# step2: 初始化filed, 读入气象数据
#        初始化 c_init
         初始化 bottom_flux
"""

"""
通过这个步骤，获得XCO2统计数据上，各层的权重系数
"""
def calculate_weight_z(vector_z):
    pressure_ratio = -vector_z/100000
    pressure_ratio_pad = torch.concat([torch.tensor([1.0]), pressure_ratio, torch.tensor([0.0])])
    left_gap = -pressure_ratio_pad[1:-1] + pressure_ratio_pad[0:-2]
    left_gap[1::] = left_gap[1::]/2.0
    right_gap = -pressure_ratio_pad[2::] + pressure_ratio_pad[1:-1]
    right_gap[0:-1] = right_gap[:-1]/2.0
    return ( left_gap + right_gap)
    
    
vector_z = vector_z_orgine[vertical_indexs]
weight_z = calculate_weight_z(vector_z)
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
         


longitude_vector, latitude_vector = get_variable_Merra2_vector_single(merra2_folder, geoschem_statemet_file, variable_list = ["lon","lat"])

# 这边注意 c_init 和 bottom_flux的来源
c_init = get_c_init(source_config = "geos-chem", folder_name=geoschem_folder, 
                    file_name = geoschem_co2_file, 
                    keep_vertical = vertical_indexs  )

bottom_flux_ct_filename = os.path.join(experiment_folder, "carbontracker/CT2019B.flux1x1.201807_reshape.npy")
bottom_flux = get_bottom_flux(source_config = "carbon_tracker", 
                              time_len = u_all.shape[0],
                              file_name = bottom_flux_ct_filename)

#%%
state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

# 讲变量转到device
variable_list = [state_all,time_vector, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor]
for variable in variable_list:
    variable = variable.to(device)

# 开始数值仿真
#%%

model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), 
    dim = sim_dimension)

model = model.to(device)

with torch.no_grad():
    print("start simulation")
    start_time = time.time()
    total_result = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top_clone, 
                                                       vertical_mapping,weight_z,
                                                       if_mixing=if_mixing)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
    
print(total_result.shape)

#%%
save_result = True
if(save_result):
    write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                 output_nc_name= os.path.join(experiment_folder, "result/simulation_result_30d_benchmark.nc"),
                 time_string=time_string, plot_interval=48,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z[0:preserve_layers])
    
    x_result = total_result*weight_z*len(weight_z)
    write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                  output_nc_name= os.path.join(experiment_folder, "result/simulation_result_30d_benchmark_x.nc"),
                  time_string=time_string, plot_interval=48,
                  longitude_vector = longitude_vector,
                  latitude_vector = latitude_vector)

"""
上面的代码和反演问题无关!!!!
"""

#%%

# 需要打成 index_vector 和 value_vector
satellite_filename = os.path.join(experiment_folder,"satellite/satellite_data_20190701_240_72_46.npz")
oco_index_vector, oco_value_vector , oco_time_vector = get_oco(satellite_filename)  
oco_value_vector_torch = torch.tensor(oco_value_vector)/10000/100.

x_con_true = torch.mean(total_result[:, 1:2, :,:,:]*weight_z*len(weight_z), dim = (1,4))
x_con_true_obs = x_con_true.flatten()[oco_index_vector].detach()


#%%
# flux_lr = 3.0e-10
init_lr = 3.0e-5
c_init_true = c_init
c_init = get_c_init(source_config = "init_constant", keep_vertical=vertical_indexs)


c_init.requires_grad = True
optimizer_cinit = Adam([c_init], lr=init_lr)

print(f"init_lr:{init_lr}")



#%%
state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

criterion_mse = torch.nn.MSELoss()
criterion_mae = torch.nn.L1Loss()

iteration_number = 100

for iteration in range(iteration_number):
    # 需要重新赋值
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)
    print(f"iteration:{iteration}")
    start_time = time.time()
    
    state_full_pred = era5_transport_simulation_3d_mixing(model, time_vector, state_all,pbl_top_clone, 
                                                       vertical_mapping,weight_z,
                                                       if_mixing=if_mixing)
    
    # 变量选择浓度, 在变量和高度方向平均
    # concentration_tensor = torch.mean(state_full_pred[:, 1:2, :,:,:], dim = (1,4))
    
    x_con_predict = torch.mean(state_full_pred[:, 1:2, :,:,:]*weight_z*len(weight_z), dim = (1,4))
    x_con_predict_obs = x_con_predict.flatten()[oco_index_vector]   
    

    true_tensor = oco_value_vector_torch
    predict_tensor = x_con_predict_obs
    
    loss = criterion_mse(predict_tensor, true_tensor)
    loss_mae = criterion_mae(predict_tensor, true_tensor)
    print( f"iteration:{iteration}, con mse loss:{loss.item():.4g}, mae:{loss_mae.item():.4g}")
    
    x_cinit = torch.mean(c_init*weight_z*len(weight_z), axis = -1)
    x_c_init_true = torch.mean(c_init_true*weight_z*len(weight_z), axis = -1)
    
    # cinit_mae = criterion_mae(x_cinit, x_c_init_true)
    # cinit_mse = criterion_mse(x_cinit, x_c_init_true)
    # print( f"iteration:{iteration}, cinit mse:{cinit_mse.item():.4g}, mae:{cinit_mae.item():.4g}")
    
    optimizer_cinit.zero_grad()
    loss.backward()
    optimizer_cinit.step()
    

    print(f"elapse time:{time.time() - start_time: .3e}")
    
    
    write_carbon_netcdf_3d_geoschem(data_= state_full_pred.detach().numpy(),
                 output_nc_name= os.path.join(experiment_folder, 
                 f"result/assimilation_result_ocoobs_3e-5_{str(iteration).zfill(3)}.nc"),
                 time_string=time_string, plot_interval=48,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z[0:preserve_layers])
    
    
    x_result = state_full_pred*weight_z*len(weight_z)
    write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                  output_nc_name= os.path.join(experiment_folder, 
                 f"result/assimilation_result_ocoobs_3e-5_x{str(iteration).zfill(3)}.nc"),
                  time_string=time_string, plot_interval=48,
                  longitude_vector = longitude_vector,
                  latitude_vector = latitude_vector)
    
    torch.save(c_init,os.path.join(experiment_folder, 
    f"result/assimilation_result_ocoobs_3e-5_cinit{str(iteration).zfill(3)}.pt"))
    
    
    
    
    
     
