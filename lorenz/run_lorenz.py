#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""
import os
import sys
sys.path.append("..") 
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
from dynamic_model.integral_module import RK4Method, JRK4Method, VJRK4Method, da_odeint, rk4_step, Jrk4_step, RK4_Method
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss, classical3dvar

from tools.kalman_tools import KF_cal_Kmatrix, KF_update_state, KF_update_corvariance, KF_cal_Bmatrix
from tools.kalman_tools import KF_cal_Bmatrix_ensemble, KF_update_state_batch, KF_update_state_corvariance_ensemble

from dynamic_model.Lorenz63 import Lorenz63, construct_lorenz63_initial_state,construct_lorenz63_cov_matrix
from dynamic_model.Lorenz63 import plot_Lorenz63_onevector, evaluate_lorenz63_laststate, plot_Lorenz63_errormap,plot_Lorenz63_trajectory


class Args:
    main_folder = "framework"   # 问题定义
    sub_folder = "Lorenz63"  # 主方案定义
    prefix = "normal_case"
    adjoint = True
    method = "rk4"
    pde_problem = "Lorenz63"  # "Lorenz63,Lorenz96,Burgers,ShallowWater"
    da_method = "auto"           # 3dvar, 4dvar, KF, enKF, auto, unchainAuto"
    total_iteration = 50  # Lorenz63: 100, Lorenz96: 100
    lr = 0.05  # Lorez63: 0.05 , Lorenz96: 0.02
    optimizer = "adam"  # "adam, norm" Lorenz63:norm,  Lorenz96:adam,
    R_inv_type = 2  # ShallowWater 为0， 其他都是 2


args = Args()
FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))

# 为了画图用
folder_name = os.path.join(
    "./results/pisr/", args.pde_problem, args.da_method)
FileUtils.makedir(folder_name)


FileUtils.makefile(os.path.join(
    "logs", args.main_folder, args.sub_folder), args.prefix + ".txt")

# set logging
filehandler = logging.FileHandler(os.path.join(
    "logs", args.main_folder, args.sub_folder, args.prefix + ".txt"))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# random variable
random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)

torch.set_printoptions(precision=10)

# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
logger.info(f"device:{device}")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True

logger.info(args)

###########   begin assimilation, initial state and model  ###########
# set problem model, and the initial state

if(args.pde_problem == "Lorenz63"):  # "Lorenz63,Lorenz96,Burgers,ShallowWater"
    print("-"*20 + "Lorenz63 Problem" + "-"*20)
    grid_info, state_info, time_info = construct_lorenz63_initial_state()

    (state_init_pred_save, state_init_true) = state_info
    (time_vector, nt_time, ind_obs, nt_obs) = time_info

    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
    pde_model = Lorenz63(sigma, beta, rho).to(device)

    R_sigma, B_sigma, R_mat, B_mat, R_inv_2, R_inv_1, R_inv_0, B_inv, state_background = construct_lorenz63_cov_matrix(
        state_init_true)



logger.info(f"model parameter:{ModelUtils.get_parameter_number(pde_model)}")
logger.info(f"state_init_true shape:{state_init_true.shape}")

time_vector = time_vector.to(device)
state_init_true = state_init_true.to(device)


# twin framework
observe_ops = pde_model.observe
with torch.no_grad():
    state_final_true, state_full_true = da_odeint(
        pde_model, state_init_true, time_vector, method=None, device=device)

    from torchdiffeq import odeint_adjoint as odeint
    state_full_true_node = odeint(pde_model, state_init_true, time_vector,
                                  method=args.method).to(device)
    print(f"final state mean daode:{torch.mean(state_full_true[-1,:])}")
    print(f"final state mean  node:{torch.mean(state_full_true_node[-1,:])}")
    print(state_full_true.device, state_full_true_node.device)
    print(
        f"simulation difference:{torch.sum(torch.abs(state_full_true[-1,:] - state_full_true_node[-1, :]))}")


# %% 准备用来做超分的数据

print(state_full_true.shape)
if (args.pde_problem == "Lorenz63"):

    folder_name = os.path.join(
        "./results/pisr/", args.pde_problem, args.da_method)
    
    filename = args.prefix + "_trajectory.png"

    plot_Lorenz63_trajectory(time_vector, state_full_true, folder_name, filename)
