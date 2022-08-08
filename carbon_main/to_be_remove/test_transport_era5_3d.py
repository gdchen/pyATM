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
import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf,write_carbon_netcdf_3d,get_variable_carbon_2d, get_variable_carbon_3d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all,construct_state_flux_one_3d


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



#%% 坐标, flux, c, u, v 时间序列

def construct_ERA5_v2_initial_state_3d():
    """
    初始化网格向量，初始化时间轴
    """
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离
    dx = (6371 + 5)*1000 * 2.0*np.pi / 360.0
    dy = dx
    dz = 20000
    x, y, u, v, w, c = get_variable_carbon_3d(folder_name, file_name, layers = 10)

    vector_x = torch.tensor(x * dx)
    vector_y = torch.tensor(y * dy)
    # vector_z = torch.tensor(torch.arange(0,3) * dz, dtype= torch.float32)
    vector_z = torch.tensor([0.0, 1000,2000,3000,4000,5500,7000,9000, 13000, 20000])

    grid_x, grid_y, grid_z = torch.meshgrid(vector_x, vector_y, vector_z)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))

    map_factor[map_factor > 3.5] = 3.5
    map_factor[map_factor < 0.0] = 3.5

    grid_info = (dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor)

    #### time info, 10分钟间隔, 7天  ####
    delta_second = 60*60
    nt_time = int(1*24*7)-1
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1)

    time_string = [datetime.datetime(2020, 9, 23) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]

    time_info = time_vector, time_string, nt_time
    return grid_info, time_info


# 初始化网格信息，以及时间信息
grid_info, time_info = construct_ERA5_v2_initial_state_3d()
time_vector, time_string, nt_time = time_info
dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor = grid_info



def add_flux_m01(time_index, f_temp):
    """
    添加源项 
    """
    if(( time_index <72 )):
        for x in range(120,140):
            for y in range(55,75):
                if((x - 130)**2 + (y - 65)**2 < 10**2):
                    f_temp[x,y,0] = f_temp[x,y,0] + 1.5e-7

    if(( time_index >= 48 ) & ( time_index <96 )):
        f_temp[50:60,60:70,0] =  f_temp[50:60,60:70,0] - 1.5e-7
    
    return f_temp





def construct_cuv_all_3d(u,v,w, c, time_vector):
    u_all_list = list()
    v_all_list = list()
    w_all_list = list()
    f_all_list = list()
    c_all_list = list()

    for time_index ,value in enumerate(time_vector):
        """
        这边要格外注意, 需要匹配对应时间
        """
        u_all_list.append(torch.tensor(u[time_index//2,:,:,:]))
        v_all_list.append(torch.tensor(v[time_index//2,:,:,:]))
        w_all_list.append(torch.tensor(w[time_index//2,:,:,:]))
        c_temp = torch.tensor(c[time_index //2,:,:,:])
        f_temp = torch.zeros( c_temp.shape)
        c_temp = torch.zeros( c_temp.shape)

        # c_temp = torch.tensor(c[time_index //2,:,:,:])
        f_temp = add_flux_m01(time_index, f_temp)

        c_all_list.append(c_temp)
        f_all_list.append(f_temp)

    u_all = torch.stack(u_all_list)
    v_all = torch.stack(v_all_list)
    w_all = torch.stack(w_all_list)
    c_all = torch.stack(c_all_list)
    f_all = torch.stack(f_all_list)

    return f_all, c_all, u_all, v_all, w_all


# 初始化filed
_, _, u, v, w, c = get_variable_carbon_3d(folder_name, file_name, layers = 10)
f_all, c_all, u_all, v_all, w_all = construct_cuv_all_3d(u,v,w, c, time_vector )
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

write_carbon_netcdf_3d(data_=total_result.detach().numpy(),
             ref_nc_name= os.path.join(folder_name, file_name),
             output_nc_name='../data/nc_file/result_simulation_3d_0209_2.nc',
             time_string=time_string, plot_interval=3, layers = 10, vector_z = vector_z)

if(False):
    exit()

case_name = "one"


if(case_name == "one"):
    flux_one = torch.zeros(u_all[0,::].shape, )
    flux_one.requires_grad = True
    mask = torch.ones(u_all.shape)
    state_all =  construct_state_flux_one_3d(flux_one, mask, c_all, u_all, v_all, w_all)



# 开始梯度更新
if(case_name == "one"):
    optimizer = Adam([flux_one], lr=1e-8)
    # optimizer = Adam([flux_one], lr=1e-10)


criterion = torch.nn.MSELoss()

print("start data assimilation")
for iteration in range(300):
    print(iteration)
    time_start = time.time()

    state_full_pred = era5_transport_simulation_3d(model, time_vector, state_all)
    """
    只监督到碳浓度上
    """
    loss = criterion(state_full_pred[:, 1:2, ::],
                     total_result[:, 1:2, ::])
    
    print(torch.mean(state_full_pred[:, 1:2, ::]), torch.mean(total_result[:, 1:2, ::]) )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

    """
    prepare for the next iteration, only update the flux. 
    由 可学习参数, 构造state
    """

    if(case_name == "one"):
        state_all = construct_state_flux_one_3d(flux_one, mask, c_all, u_all, v_all,w_all)

#    一天更新一个通量
#    if(case_name == "daily"):
    



    write_carbon_netcdf_3d(data_=state_full_pred.detach().numpy(),
             ref_nc_name="../data/nc_file/myfile_carbon.nc",
             output_nc_name=f'../data/nc_file/result_simulation_result_3d_{iteration}.nc',
             time_string=time_string, plot_interval=3)



