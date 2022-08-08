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
import torch
import time
from dynamic_model.Transport import CTM_Model_3D
from dynamic_model.Transport import ctm_simulation_3d_mixing
from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem
from autograd_utils.carbon_tools import construct_state_with_cinitfluxbottom
from autograd_utils.carbon_tools  import get_bottom_flux, get_c_init
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 
from autograd_utils.geoschem_tools import construct_uvw_all_3d,generate_vertical_info,regenerate_vertcal_state
from autograd_utils.geoschem_tools  import calculate_x_result

class Args:
    year = 2019
    month = 7
    day = 1
    last_day = 5       # 仿真天数
    interval_minutes = 30    # 仿真时间间隔，需要与merra2文件夹下的结果保持一致。
    device = "cpu"
    if_mixing = True  # 是否开启边界层混合 
    sim_dimension = 3  #  2,3: 二维还是3维
    layer_type = "layer_9" # 仿真的层数配置 "layer_9","layer_47","layer_1"
    if_plot_result = True     # 是否需要保存仿真结果
    plot_interval = 48        # 每多少个时间步保存一份结果, 30min的仿真，对应的是1天
    result_prefix = "torch_ctm" # 输出文件的前缀
    experiment_folder= '/Users/yaoyichen/dataset/auto_experiment/experiment_0/'
    geoschem_co2_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"


args = Args()
simulation_len = int(args.last_day*1440//args.interval_minutes)
merra2_folder = os.path.join(args.experiment_folder, 'merra2')
geoschem_folder = os.path.join(args.experiment_folder,  'geoschem')

# step1.1 读入气象数据
print("loading merra2 data...")
startswith_string = "GEOSChem.StateMet."
u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine, merra_file_name_list \
    = construct_uvw_all_3d( merra2_folder, startswith_string,
                           args)

#%%
# step1.2 初始化网格信息，以及时间信息
grid_info, time_info = construct_Merra2_initial_state_3d(folder_name = merra2_folder,
                                                         file_name = merra_file_name_list[0],
                                                         year = args.year,
                                                         month = args.month,
                                                         day = args.day,
                                                         last_day = args.last_day,
                                                         interval_minutes = args.interval_minutes)

time_vector, time_string, nt_time = time_info
(longitude_vector,latitude_vector,
             dx, dy, dz, _1, _2, _3, 
             vector_x, vector_y, vector_z_origine, map_factor_origine) = grid_info

#%%
# step 1.3 配置高度方向信息，生成特定高度上的状态
vertical_mapping = generate_vertical_info(args.layer_type)
         
[vector_z, map_factor, u_all, v_all, w_all], weight_z, pbl_top \
    = regenerate_vertcal_state(vector_z_origine, map_factor_origine, 
                             u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine, vertical_mapping)


# step 1.4 初始化CO2浓度信息， 初始化bottom-up flux 
c_init = get_c_init(source_config = "geos-chem", folder_name=geoschem_folder, 
                    file_name = args.geoschem_co2_file, 
                    keep_vertical = vertical_mapping["vertical_indexs"]  )


bottom_flux = get_bottom_flux(source_config = "guodong", 
                              time_len = u_all.shape[0],
                              file_name = os.path.join(args.experiment_folder, 
                                                      
                                                       
# bottom_flux = get_bottom_flux(source_config = "carbon_tracker", 
#                               time_len = u_all.shape[0],
#                               file_name = os.path.join(args.experiment_folder, 
#                                                         "carbontracker/CT2019B.flux1x1.201807_reshape.npy"))

# bottom_flux = get_bottom_flux(source_config = "init_constant", 
#                               time_len = u_all.shape[0])

#%% 
## step 1.5 将 c_init, bottom_flux, u_all, v_all,w_all 配置成大向量 state_all
#生成 state_all: [time, variable, longitude, latitude, height]
# variable 排列 flux, con, u, v, w
# state_all.shape: torch.Size([240, 5, 72, 46, 9])
# c_init.shape: torch.Size([1, 72, 46, 9])
# bottom_flux.shape: torch.Size([240, 72, 46, 1])

state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

variable_list = [state_all,time_vector, vector_x, vector_y, vector_z, map_factor]
for variable in variable_list:
    variable = variable.to(args.device)


#%% 开始数值仿真
# step 2.1 构建CTM 模型
model = CTM_Model_3D(grid_info=( vector_x, vector_y, vector_z, map_factor), 
    dim = args.sim_dimension)
model = model.to(args.device)


# step 2.2 开始数值仿真
with torch.no_grad():
    print("start simulation")
    start_time = time.time()
    total_result = ctm_simulation_3d_mixing(model, time_vector, state_all,pbl_top, 
                                                       vertical_mapping, weight_z,
                                                       if_mixing = args.if_mixing)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
print(total_result.shape)

    

#%% step 3.1 写文件
if(args.if_plot_result):
    write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                 output_nc_name= os.path.join(args.experiment_folder, f"result/{args.result_prefix}_3d.nc"),
                 time_string=time_string, plot_interval= args.plot_interval,
                 longitude_vector = longitude_vector,
                 latitude_vector = latitude_vector,
                 vector_z = vector_z)
    
    # 计算高度方向加权后的X结果, 输出2维nc文件
    x_result = calculate_x_result(total_result, weight_z)
    write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                  output_nc_name= os.path.join(args.experiment_folder, f"result/{args.result_prefix}_2d.nc"),
                  time_string=time_string, plot_interval= args.plot_interval,
                  longitude_vector = longitude_vector,
                  latitude_vector = latitude_vector)


#%%

    
