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
from dynamic_model.Transport import Basic_Transport_Model, transport_simulation,ERA5_transport_Model,era5_transport_simulation
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
import datetime
from tools.netcdf_helper  import write_2d_carbon_netcdf, get_variable_carbon_2d
from tools.carbon_tools import  construct_state_flux_one,construct_state_flux_seven, construct_state_flux_all

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


#%% 坐标, flux, c, u, v 时间序列
# (dx, dy, grid_x, grid_y, vector_x,
#  vector_y), state0 = construct_initial_state()
# print(state0.shape, grid_x.shape)
# time_vector = torch.linspace(0., 1.0, 101)
# f_all, c_all, u_all, v_all = construct_cuv_all(state0, time_vector)
# state_all = torch.stack([f_all, c_all, u_all, v_all]).permute(1,0,2,3)

longitude, latitude, u, v, c = get_variable_carbon_2d("a", "b")


def construct_ERA5_v2_initial_state():
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离
    dx = (6371 + 5)*1000 * 2.0*np.pi / 360.0
    dy = dx
    x, y, u, v, c = get_variable_carbon_2d("a", "b")

    # print(f"x:{x} ,y:{y}")
    vector_x = torch.tensor(x * dx)
    vector_y = torch.tensor(y * dy)
    grid_x, grid_y = torch.meshgrid(vector_x, vector_y)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))

    map_factor[map_factor > 3.5] = 3.5
    map_factor[map_factor < 0.0] = 3.5


    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor)

    #### time info, 10分钟间隔, 7天  ####
    delta_second = 60*60
    nt_time = int(1*24*7)-1
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1)

    time_string = [datetime.datetime(2020, 9, 23) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]

    time_info = time_vector, time_string, nt_time
    return grid_info, time_info

grid_info, time_info = construct_ERA5_v2_initial_state()
time_vector, time_string, nt_time = time_info
dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor = grid_info


x, y, u, v, c = get_variable_carbon_2d("a", "b")


def construct_cuv_all(u,v,c, time_vector):
    u_all_list = list()
    v_all_list = list()
    f_all_list = list()
    c_all_list = list()

    for i,value in enumerate(time_vector):
        """
        这边要格外注意, 需要匹配对应时间
        """
        u_all_list.append(torch.tensor(u[i//2,:,:]))
        v_all_list.append(torch.tensor(v[i//2,:,:]))
        c_temp = torch.tensor(c[i//2,:,:])
        f_temp = torch.zeros( c_temp.shape)

        """
        添加源项
        """
        if((i<72 )):
            for x in range(120,140):
                for y in range(55,75):
                    if((x - 130)**2 + (y - 65)**2 < 10**2):
                        f_temp[x,y] = f_temp[x,y] + 1.5e-7

        if((i>= 48 ) & (i<96 )):
            f_temp[50:60,60:70] =  f_temp[50:60,60:70] - 1.5e-7

        c_all_list.append(c_temp)
        f_all_list.append(f_temp)

    u_all = torch.stack(u_all_list)
    v_all = torch.stack(v_all_list)
    c_all = torch.stack(c_all_list)
    f_all = torch.stack(f_all_list)

    return f_all, c_all, u_all, v_all

f_all, c_all, u_all, v_all = construct_cuv_all(u,v,c, time_vector )
state_all = torch.stack([f_all, c_all, u_all, v_all]).permute([1,0,2,3])

model = ERA5_transport_Model(grid_info=(
    dx, dy, grid_x, grid_y, vector_x, vector_y,map_factor))

print(state_all.shape)
with torch.no_grad():
    total_result = era5_transport_simulation(model, time_vector, state_all)


write_2d_carbon_netcdf(data_=total_result.detach().numpy(),
             ref_nc_name="../data/nc_file/myfile_carbon.nc",
             output_nc_name='../data/nc_file/result_simulation_addsource_new.nc',
             time_string=time_string, plot_interval=3)

exit()

# 读入模式开始更新
"""
以下为数据同化部分
开始先配置设置
"""
case_name = "seven"


if(case_name == "seven"):
    # 获得同化的初始场
    flux1 = torch.zeros(u_all[0,::].shape, )
    flux1.requires_grad = True
    flux2 = torch.zeros(u_all[0,::].shape, )
    flux2.requires_grad = True
    flux3 = torch.zeros(u_all[0,::].shape, )
    flux3.requires_grad = True
    flux4 = torch.zeros(u_all[0,::].shape, )
    flux4.requires_grad = True
    flux5 = torch.zeros(u_all[0,::].shape, )
    flux5.requires_grad = True
    flux6 = torch.zeros(u_all[0,::].shape, )
    flux6.requires_grad = True
    flux7 = torch.zeros(u_all[0,::].shape, )
    flux7.requires_grad = True

    mask = torch.ones(u_all.shape)

    """
    c_all.requires_grad = False
    u_all.requires_grad = False
    v_all.requires_grad = False
    """

    flux_list = [flux1,flux2,flux3,flux4,flux5,flux6,flux7 ]
    state_all = construct_state_flux_seven(flux_list, mask, c_all, u_all, v_all)

elif(case_name == "all"):
    flux_all = torch.zeros(u_all.shape, )
    flux_all.requires_grad = True
    mask = torch.ones(u_all.shape)
    state_all =  construct_state_flux_all(flux_all, mask, c_all, u_all, v_all)

elif(case_name == "one"):
    flux_one = torch.zeros(u_all[0,::].shape, )
    flux_one.requires_grad = True
    mask = torch.ones(u_all.shape)
    state_all =  construct_state_flux_one(flux_one, mask, c_all, u_all, v_all)



# 开始梯度更新
if(case_name == "seven"):
    """
    这个量级，和flux本身的量级很低有关系
    """
    optimizer = Adam([flux1,flux2,flux3,flux4,flux5,flux6,flux7 ], lr=1e-8)
elif(case_name == "all"):
    optimizer = Adam([flux_all], lr=1e-8)

elif(case_name == "one"):
    optimizer = Adam([flux_one], lr=1e-8)


criterion = torch.nn.MSELoss()

print("start data assimilation")
for iteration in range(300):
    print(iteration)
    time_start = time.time()

    state_full_pred = era5_transport_simulation(model, time_vector, state_all)
    """
    只监督到碳浓度上
    """
    loss = criterion(state_full_pred[:, 1:2, :, :],
                     total_result[:, 1:2, :, :])
    
    print(torch.mean(state_full_pred[:, 1:2, :, :]), torch.mean(total_result[:, 1:2, :, :]) )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

    """
    prepare for the next iteration, only update the flux. 
    由 可学习参数, 构造state
    """
    # 
    if(case_name == "seven"):
        flux_list = [flux1,flux2,flux3,flux4,flux5,flux6,flux7 ]
        state_all = construct_state_flux_seven(flux_list, mask, c_all, u_all, v_all)

    elif(case_name == "all"):
        state_all =  construct_state_flux_all(flux_all, mask, c_all, u_all, v_all)

    elif(case_name == "one"):
        state_all = construct_state_flux_one(flux_one, mask, c_all, u_all, v_all)



    write_2d_carbon_netcdf(data_=state_full_pred.detach().numpy(),
             ref_nc_name="../data/nc_file/myfile_carbon.nc",
             output_nc_name=f'../data/nc_file/result_simulation_result_{iteration}.nc',
             time_string=time_string, plot_interval=3)

