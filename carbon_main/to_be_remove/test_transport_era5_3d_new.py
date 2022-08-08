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

from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco
from dynamic_model.ERA5_v2 import filter_longitude, filter_latitude


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
preserve_layers = 3


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
    x, y, u, v, w = get_uvw()

    vector_x = torch.tensor(x * dx)
    vector_y = torch.tensor(y * dy)
    vector_z = torch.tensor([0.0, 1000,2000,3000,4000,5500,7000,9000, 13000, 20000])[0:preserve_layers]

    grid_x, grid_y, grid_z = torch.meshgrid(vector_x, vector_y, vector_z)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))

    map_factor[map_factor > 2.5] = 2.5
    map_factor[map_factor < 0.0] = 2.5

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



def add_flux_m01(f_in):
    """
    增加源项 flux 的方式1, 增加一个圈和一个方块
    """
    for time_index in range(len(f_in)):

        if(( time_index <72 )):
            for x in range(120,140):
                for y in range(55,75):
                    if((x - 130)**2 + (y - 65)**2 < 10**2):
                        f_in[time_index, x,y,0] = f_in[time_index, x,y,0] + 0.5e-10

        if(( time_index >= 48 ) & ( time_index <96 )):
            f_in[time_index, 50:60,60:70,0] =  f_in[time_index, 50:60,60:70,0] - 0.5e-10
        
    return f_in


def add_flux_m03(f_in):
    """
    增加源项 flux 的方式1, 增加一个圈和一个方块
    """
    for time_index in range(len(f_in)):
        f_in[time_index, 55:58,67:71,0] =  f_in[time_index, 55:58,67:71,0] + 0.5e-10
    return f_in


def add_flux_m02(f_in):
    """
    增加源项 flux 的方式2, 读入bottom-up 方案产生的flux 
    """
    f_add = get_f()
    f_in[:,:,:,0] = torch.tensor(f_add)*10e-3*0.8/10.

    return f_in


def construct_cuv_all_3d():
    _, _, u, v, w = get_uvw()
    """
    w方向速度降为很小很小
    """
    u_all = torch.tensor(u, dtype= torch.float32)[:,:,:,0:preserve_layers]
    v_all = torch.tensor(v, dtype= torch.float32)[:,:,:,0:preserve_layers]
    w_all = torch.tensor(w, dtype= torch.float32)[:,:,:,0:preserve_layers]/100.

    # for i in range(1):
    #     tt = w_all[:,:,:,0]
    #     print(tt.shape)
    #     tt = filter_longitude(tt, nx  = tt[0], ny = tt[1], percentage= 4 )
    #     print(tt.shape)
    #     tt = filter_latitude(tt, nx  = tt[0], ny = tt[1], percentage= 4 )
    #     print(tt.shape)
    # print(w_all.shape)

    # w_all = filter_longitude(w_all, nx  = w_all.shape[1], ny = w_all.shape[2], percentage= 2 )
    # print(w_all.shape)
    # w_all = filter_latitude(w_all, nx  = w_all.shape[1], ny = w_all.shape[2], percentage= 2 )

    # print(w_all.shape)


    c = get_c()
    
    # c = get_c_zero(u_all)
    c_all = torch.tensor(c, dtype= torch.float32).repeat([len(u_all),1,1,1])[:,:,:,0:preserve_layers]

    f_all = torch.zeros(c_all.shape, dtype= torch.float32)[:,:,:,0:preserve_layers]

    f_all  = add_flux_m02(f_all)

    return f_all, c_all, u_all, v_all, w_all


# 初始化filed
f_all, c_all, u_all, v_all, w_all = construct_cuv_all_3d( )
state_all = torch.stack([f_all, c_all, u_all, v_all, w_all]).permute([1,0,2,3,4])

print(state_all.shape)

model = ERA5_transport_Model_3D(grid_info=(
    dx, dy, dz, grid_x, grid_y, grid_z, vector_x, vector_y, vector_z, map_factor), dim = 3)


# 开始数值仿真
with torch.no_grad():
    print("start simulation")
    start_time = time.time()
    total_result = era5_transport_simulation_3d(model, time_vector, state_all)
    print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")

print(total_result.shape)

write_carbon_netcdf_3d(data_=total_result.detach().numpy(),
             ref_nc_name= os.path.join(folder_name, file_name),
             output_nc_name='../data/nc_file/result_simulation_3d_0212_oco_temp.nc',
             time_string=time_string, plot_interval=3, layers = preserve_layers, vector_z = vector_z[0:preserve_layers])


# total_result[-1,1:2,:,:,:] = c_all[0,:,:,:]
if(False):
    exit()

"""
如果flux恒定且不随时间变化
"""
case_name = "one"

if(case_name == "one"):
    flux_one = torch.zeros(u_all[0,::].shape, )
    flux_one.requires_grad = True
    mask = torch.zeros(u_all.shape)
    mask[:,:,:,0] = 1

    """
    一開始浓度值为平均值
    """
    # print(c_all.shape)
    # c_all[0,:,:,0] = torch.mean(c_all[0,:,:,0] , dim = (0,1))
    # c_all[0,:,:,1] = torch.mean(c_all[0,:,:,1] , dim = (0,1))
    # c_all[0,:,:,2] = torch.mean(c_all[0,:,:,2] , dim = (0,1))
    c_all[::] = 0.000410

    state_all =  construct_state_flux_one_3d(flux_one, mask, c_all, u_all, v_all, w_all)
 

# 开始梯度更新
if(case_name == "one"):
    # twin framework 下的反演用较小的梯度 0.3e-11
    optimizer = Adam([flux_one], lr=0.5e-11)
    # optimizer = Adam([flux_one], lr=3e-11)


criterion = torch.nn.MSELoss()

print("start data assimilation")
start_time = time.time()

oco_index_vector, oco_value_vector = get_oco()
oco_value_vector_torch = torch.tensor(oco_value_vector)

for iteration in range(300):
    print(iteration)
    time_start = time.time()

    state_full_pred = era5_transport_simulation_3d(model, time_vector, state_all)

    concentration_tensor = torch.mean(state_full_pred[:, 1:2, :,:,:], dim = (1,4))
    print("#"*30)
    print(concentration_tensor.shape)
    concentration_tensor_observe = concentration_tensor.flatten()[oco_index_vector]   
    loss = criterion(concentration_tensor_observe, oco_value_vector_torch)
    print(torch.mean(concentration_tensor_observe), torch.mean(oco_value_vector_torch) )
    """
    只监督到碳浓度上
    """
    
    # loss = criterion(state_full_pred[:, 1:2, ::],
    #                  total_result[:, 1:2, ::])
    
    # print(torch.mean(state_full_pred[:, 1:2, ::]), torch.mean(total_result[:, 1:2, ::]) )
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

    write_carbon_netcdf_3d(data_= state_full_pred.detach().numpy(),
             ref_nc_name="../data/nc_file/myfile_carbon.nc",
             output_nc_name=f'../data/nc_file/result_simulation_result_3d_0210_full_ocozero_{iteration}.nc',
             time_string=time_string, plot_interval=3, layers = preserve_layers, vector_z = vector_z[0:preserve_layers])

    print(f"elapse time:{time.time() - start_time}")



