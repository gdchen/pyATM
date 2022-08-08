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
from dynamic_model.integral_module import RK4Method, JRK4Method, VJRK4Method, da_odeint, rk4_step, Jrk4_step, RK4_Method, da_odeint_boundary
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss, classical3dvar
from tools.plot_tools import plot_2d
from dynamic_model.ERA5_v2 import ERA5_pressure, construct_ERA5_v2_initial_state
from dynamic_model.ERA5_v2 import write_netcdf, filter_latitude, filter_longitude

grid_info, state_info, time_info = construct_ERA5_v2_initial_state()

(time_vector, time_string, nt_time, ind_obs, nt_obs) = time_info
dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor = grid_info
(state, state0_true) = state_info

# scale
# state[2, :, :] = state[2, :, :] / 1000.0
# state0_true[2, :, :] = state0_true[2, :, :]/1000.0
print(state0_true.shape)

pde_model = ERA5_pressure(grid_info)

device = "cpu"
obs_count = 0


def simulation(time_vector, init_state):
    total_result = torch.empty(tuple(time_vector.shape,) + init_state.shape)
    total_result[0, :] = init_state

    state = init_state
    for index, time in enumerate(time_vector[1::]):
        # print(index, state.mean(dim=(1, 2)))
        dt = time_vector[index + 1] - time_vector[index]
        state = state + \
            rk4_step(pde_model, time, dt, state, device=device)

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


with torch.no_grad():
    state_full_true = simulation(time_vector, state0_true)
    state_full_false = simulation(time_vector, state)

    write_netcdf(data_=state_full_true.detach().numpy(),
                 ref_nc_name="/Users/yaoyichen/Desktop/nc_file/myfile19.nc",
                 output_nc_name='/Users/yaoyichen/Desktop/nc_file/result_da_true.nc',
                 time_string=time_string, plot_interval=36)

    write_netcdf(data_=state_full_false.detach().numpy(),
                 ref_nc_name="/Users/yaoyichen/Desktop/nc_file/myfile19.nc",
                 output_nc_name='/Users/yaoyichen/Desktop/nc_file/result_da_false.nc',
                 time_string=time_string, plot_interval=36)


def calculate_error(state_full_pred, state_full_true):
    with torch.no_grad():
        u_mean = torch.mean(
            torch.abs(state_full_pred[0, 0, :, :] - state_full_true[0, 0, :, :]))
        v_mean = torch.mean(
            torch.abs(state_full_pred[0, 1, :, :] - state_full_true[0, 1, :, :]))
        phi_mean = torch.mean(
            torch.abs(state_full_pred[0, 2, :, :] - state_full_true[0, 2, :, :]))
    return u_mean, v_mean, phi_mean


state.requires_grad = True
optimizer = Adam([state], lr=0.3)

for iteration in range(200):
    time_start = time.time()
    state_full_pred = simulation(time_vector, state)

    criterion = torch.nn.MSELoss()
    loss = criterion(state_full_pred[ind_obs, 0:1, :, :],
                     state_full_true[ind_obs, 0:1, :, :])
    + criterion(state_full_pred[ind_obs, 1:2, :, :],
                state_full_true[ind_obs, 1:2, :, :])
    + criterion(state_full_pred[ind_obs, 2:3, :, :],
                state_full_true[ind_obs, 2:3, :, :])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    u_mean, v_mean, phi_mean = calculate_error(
        state_full_pred, state_full_true)

    print(u_mean, v_mean, phi_mean)
    print("elapse time:{}".format(time.time() - time_start))

    write_netcdf(data_=state_full_pred.detach().numpy(),
                 ref_nc_name="/Users/yaoyichen/Desktop/nc_file/myfile19.nc",
                 output_nc_name='/Users/yaoyichen/Desktop/nc_file/result_da_scale' +
                 str(iteration).zfill(2) + '.nc',
                 time_string=time_string, plot_interval=36)


# %%
