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

from tools.kalman_tools import KF_cal_Kmatrix, KF_update_state, KF_update_corvariance, KF_cal_Bmatrix
from tools.kalman_tools import KF_cal_Bmatrix_ensemble, KF_update_state_batch, KF_update_state_corvariance_ensemble

from dynamic_model.ERA5 import getUV_height5, construct_ERA5_initial_state, ERA5_basic, plot_ERA5


class Args:
    main_folder = "framework"   # 问题定义
    sub_folder = "ERA5"  # 主方案定义
    prefix = "normal_case"
    # adjoint = True
    # method = "rk4"
    pde_problem = "ERA5"  # "Lorenz63,Lorenz96,Burgers,ShallowWater"
    # da_method = "4dvar"           # 3dvar, 4dvar, KF, enKF, auto, unchainAuto"
    # total_iteration = 50  # Lorenz63: 100, Lorenz96: 100
    # lr = 0.05  # Lorez63: 0.05 , Lorenz96: 0.02
    # optimizer = "adam"  # "adam, norm" Lorenz63:norm,  Lorenz96:adam,
    # R_inv_type = 2  # ShallowWater 为0， 其他都是 2


args = Args()
FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))

# 为了画图用
folder_name = os.path.join(
    "./results/framework/", args.pde_problem)
FileUtils.makedir(folder_name)


inputfoldername = "/Users/yaoyichen/project_earth/dataset/era5_one_day/uv"
datestr = "20190101"

result = getUV_height5(inputfoldername, datestr)

### read basic info  ###
grid_info, state_info, time_info, state_full_info = construct_ERA5_initial_state()
dx, dy, grid_x, grid_y, vector_x, vector_y = grid_info
state_init_true = state_info
(time_vector, nt_time, ind_obs, nt_obs) = time_info


filename = "result_v.png"
plot_ERA5(state_full_info[0, 0, :],
          state_full_info[1, 0, :], folder_name, filename)


### read_model ###
pde_model = ERA5_basic(grid_info=grid_info)


print(state_init_true.shape)
print(state_info.shape, state_full_info.shape)


state_final_model, state_full_model = da_odeint_boundary(
    pde_model, state_init_true, time_vector, method=None)
print(state_final_model.shape, state_full_model.shape)


print(state_full_model[ind_obs].shape)
# %%
for i in range(24):
    filename = "result_u" + str(i).zfill(3) + ".png"
    plot_ERA5(state_full_info[i, 0, :],
              state_full_model[ind_obs][i, 0, :], folder_name, filename)

    filename = "result_v" + str(i).zfill(3) + ".png"
    plot_ERA5(state_full_info[i, 1, :],
              state_full_model[ind_obs][i, 1, :], folder_name, filename)
