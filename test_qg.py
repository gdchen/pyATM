#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""
import os
import sys
import numpy as np
import torch
from torch.autograd.functional import jacobian
from torch.autograd import grad
import torch.nn as nn
from torch.optim import LBFGS, Adam, SGD
from torch.distributions.multivariate_normal import MultivariateNormal
import logging
import time
import matplotlib.pyplot as plt
from dynamic_model.integral_module import RK4Method, JRK4Method, VJRK4Method, da_odeint, rk4_step, Jrk4_step, RK4_Method, da_odeint_boundary, euler_step
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss, classical3dvar
from tools.plot_tools import plot_2d
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state, construct_ERA5_v2_initial_state_1day
from dynamic_model.ERA5_v2 import write_netcdf, filter_latitude, filter_longitude

grid_info, state_info, time_info = construct_ERA5_v2_initial_state()
time_vector, time_string, nt_time, ind_obs, nt_obs = time_info
dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor = grid_info
(state, state0_true) = state_info
print(state.shape)

pde_model = ERA5_pressure(grid_info)

device = "cpu"
obs_count = 0

# total_result = torch.empty(tuple(time_vector.shape,) + state0_true.shape)
# total_result[0, :] = state0_true
# state = state0_true


def simulation(time_vector, init_state):
    total_result = torch.empty(tuple(time_vector.shape,) + init_state.shape)
    total_result[0, :] = init_state

    state = init_state
    for index, time in enumerate(time_vector[1::]):
        print(index, state.mean(dim=(1, 2)))
        dt = time_vector[index + 1] - time_vector[index]
        state = state + \
            rk4_step(pde_model, time, dt, state)

        # filter
        if index < 2:
            percentage = 10
        else:
            percentage = 4
            
        state = filter_longitude(state, nx=180, ny=90, percentage=percentage)
        state = filter_latitude(state, nx=180, ny=90, percentage=percentage)

        # boundary condition of velocity and height
        state[0, :, 0] = 0.0
        state[0, :, -1] = 0.0

        state[1, :, 0] = 0.0
        state[1, :, -1] = 0.0

        state[2, :, 0] = torch.mean(state[2, :, 1], dim=(0))
        state[2, :, -1] = torch.mean(state[2, :, -2], dim=(0))

        total_result[index + 1, :] = state

    return total_result


print(time_vector)
total_result = simulation(time_vector, state0_true)

write_netcdf(data_=total_result.detach().numpy(),
             ref_nc_name= "/Users/yaoyichen/dataset/era5/old/myfile19.nc",
             output_nc_name='./data/nc_file/result_simulation_final_0406.nc',
             time_string=time_string, plot_interval=12)

# %%def construct_ERA5_v2_initial_state_1day():

u_true, v_true, phi_true = construct_ERA5_v2_initial_state_1day()

print(np.std(phi_true - total_result[144,-1,:,:].numpy() ))
print(np.std(u_true - total_result[144,0,:,:].numpy() ))
print(np.std(v_true - total_result[144,1,:,:].numpy() ))