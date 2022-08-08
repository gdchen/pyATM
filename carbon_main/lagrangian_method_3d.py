#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:45:32 2022

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
import datetime

from dynamic_model.initialize_tools import gkern
from dynamic_model.Transport import transport_simulation,ERA5_transport_Model, ERA5_transport_Model_3D,era5_transport_simulation_3d_mixing
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude


from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.netcdf_helper import write_netcdf_single_variable_single_time_2d, write_netcdf_single_variable_multi_time_2d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d

from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw
from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco

from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem, write_carbon_netcdf_2d_avg
from autograd_utils.carbon_tools import construct_state_with_cinit,construct_state_with_fluxbottom, construct_state_with_cinitfluxbottom, construct_state_with_cinitfluxbottom_flux1time
from autograd_utils.carbon_tools  import get_bottom_flux, get_c_init
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 

from lagragian_utils.lagragian_utils import initialize_points, particle_forward_order2,generate_loc_matrix_target, generate_loc_matrix_source_target
from lagragian_utils.lagragian_utils import generate_forward_matrix_from_locmaxtrix, generate_jacobian_matrix_from_locmaxtrix
from lagragian_utils.lagragian_utils import lagragian_save, lagragian_load, plot_lagrangian_snapshot,plot_lagrangian_snapshot_basemap, plot_lagrangian_trace_basemap,plot_grid_lagrangian_trace

def construct_lagragian_uvw_all_3d(merra2_folder, startswith_string):
    """
    读入气象数据， 生成u,v,w场， 以及边界层信息
    """
    # folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    # folder_name = "/home/eason.yyc/data/carbon_inversion/201907/merra2"
    
    preserve_layers = 47
    uvw_vector =  get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"],
                                               preserve_layers = preserve_layers )

    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]
    pbl_top = get_variable_Merra2_3d_batch(merra2_folder, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = ["Met_PBLTOPL",])

    return u,v,w,pbl_top

merra2_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
startswith_string = "GEOSChem.StateMet.201907"
u_all, v_all, w_all, pbl_top = construct_lagragian_uvw_all_3d(merra2_folder, startswith_string )

#%%
"""
从经纬度坐标系
每个颗粒必须有的信息, [index, 原始的[时间index, 经度index，维度index], ]
"""
geoschem_folder = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
geoschem_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
x_grid_1d, y_grid_1d = get_variable_Merra2_vector_single(geoschem_folder, geoschem_file,variable_list = ["lon","lat"])

#%% 总共的网格划分数量 72* 44 

long_spacing = 5
lati_spacing = 4
long_block_number = int(360/long_spacing)
lati_block_number = int(176/lati_spacing)
particle_inblock = 1

simulation_minutes = 20
lagragian_hours = 3.0 
simualtion_day_span = 6

time_len = int(simualtion_day_span*24/lagragian_hours)

time_direction = "backward"   # "forward" , "backward"

output_folder_name = "./plot_folder/backward_1000/"
plot_snapshot = False
x_coordinate, y_coordinate = initialize_points(long_block_number, lati_block_number, long_spacing, lati_spacing, particle_inblock= particle_inblock)

#%%
x_point  = x_coordinate[:,:,:].flatten()
y_point  = y_coordinate[:,:,:].flatten()

x_point_all = np.zeros( [time_len,len(x_point)] )
y_point_all = np.zeros( [time_len,len(x_point)] )  

"""
反向： 时间读取从后往前读取，轨迹更新时，符号取负号
"""
print(f"start simulation")
start_time = time.time()
for time_index in range(time_len):
    print(time_index)
    if(time_direction == "forward"):
        delta_day = 1/24.0 * lagragian_hours
        merra2_time_index = int(time_index*lagragian_hours*60/simulation_minutes)
    elif(time_direction == "backward"):
        delta_day = -1.0* 1/24.0 * lagragian_hours
        merra2_time_index = int((time_len - time_index)*lagragian_hours*60/simulation_minutes) -1
    print(merra2_time_index)
    u_current = u_all[merra2_time_index,:,:,0]
    v_current = v_all[merra2_time_index,:,:,0]
    
    x_point, y_point = particle_forward_order2(x_point, y_point, u_current, v_current, 
                                               x_grid_1d, y_grid_1d,
                                               day_number = delta_day)
    
    x_point_all[time_index, :] = x_point
    y_point_all[time_index, :] = y_point
    
    if(plot_snapshot):
        plot_lagrangian_snapshot_basemap(x_point_all[time_index,:], 
                                 y_point_all[time_index,:], 
                                 os.path.join(output_folder_name, f"snapshot_{time_direction}_{str(time_index).zfill(2)}.png"))
print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")

#%% 统计每个block出现的个数
loc_matrix_target = generate_loc_matrix_target(x_point, y_point,long_spacing, lati_spacing)
loc_matrix_source_target = generate_loc_matrix_source_target(x_point, y_point,long_spacing, lati_spacing)

#%% 根据统计获得的 loc_matrix_source_target, 来绘制 jacobian_matrix_batch, forward_matrix_batch

jacobian_matrix_batch, time_vector_list = generate_jacobian_matrix_from_locmaxtrix(loc_matrix_source_target)
write_netcdf_single_variable_multi_time_2d(jacobian_matrix_batch, time_vector_list, os.path.join(output_folder_name, f"{time_direction}_jacobian_batch_{particle_inblock}.nc"), x_grid_1d, y_grid_1d[1:-1])
        
forward_matrix_batch, time_vector_list = generate_forward_matrix_from_locmaxtrix(loc_matrix_source_target)
write_netcdf_single_variable_multi_time_2d(forward_matrix_batch, time_vector_list, os.path.join(output_folder_name, f"{time_direction}_forward_batch_{particle_inblock}.nc"), x_grid_1d, y_grid_1d[1:-1])
        
#%%
sample_number = 1000
plot_lagrangian_trace_basemap(x_point_all, y_point_all, os.path.join(output_folder_name, f"trace_{time_direction}_random.png"),sample_number)


#%%

file_name = os.path.join(output_folder_name, f"lagragian{particle_inblock}.npy")
lagragian_save(file_name, x_point_all, y_point_all, loc_matrix_source_target )

x_point_all, y_point_all, loc_matrix_source_target  = lagragian_load(file_name )


#%%
#画单个grid的轨迹
long_index, lati_index = 50, 30
sample_number = 100
grid_shape = [time_len,long_block_number,lati_block_number, particle_inblock ]
file_name = os.path.join(output_folder_name, f"trace_{str(long_index).zfill(2)}_{str(lati_index).zfill(2)}.png")
plot_grid_lagrangian_trace(x_point_all, y_point_all, long_index, lati_index, file_name, grid_shape, sample_number)
