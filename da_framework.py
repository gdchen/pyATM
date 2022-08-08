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
from dynamic_model.integral_module import RK4Method, JRK4Method, VJRK4Method, da_odeint, rk4_step, Jrk4_step, RK4_Method
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss, classical3dvar

from tools.kalman_tools import KF_cal_Kmatrix, KF_update_state, KF_update_corvariance, KF_cal_Bmatrix
from tools.kalman_tools import KF_cal_Bmatrix_ensemble, KF_update_state_batch, KF_update_state_corvariance_ensemble


from dynamic_model.Lorenz63 import construct_lorenz63_initial_state, Lorenz63
from dynamic_model.Lorenz63 import plot_Lorenz63_onevector, evaluate_lorenz63_laststate, plot_Lorenz63_errormap, construct_lorenz63_cov_matrix

from dynamic_model.Lorenz96 import construct_lorenz96_initial_state, Lorenz96
from dynamic_model.Lorenz96 import plot_Lorenz96_onevector, evaluate_lorenz96_laststate, plot_Lorenz96_errormap, construct_lorenz96_cov_matrix

from dynamic_model.Burgers import construct_burgers_initial_state, Burgers
from dynamic_model.Burgers import construct_burgers_cov_matrix, plot_Burgers_onevector, plot_Burgers_errormap


from dynamic_model.ShallowWater import construct_shallowwater_initial_state, ShallowWaterModel
from dynamic_model.ShallowWater import construct_shallowwater_initial_state, construct_shallowwater_cov_matrix, construct_shallowwater_cov_matrix_complex, plot_ShallowWater_laststate, plot_ShallowWater_fullerrormap


class Args:
    main_folder = "framework"   # 问题定义
    sub_folder = "Burgers"  # 主方案定义
    prefix = "normal_case"
    adjoint = True
    method = "rk4"
    pde_problem = "ShallowWater"  # "Lorenz63,Lorenz96,Burgers,ShallowWater"
    da_method = "auto"           # 3dvar, 4dvar, KF, enKF, auto, unchainAuto"
    total_iteration = 5  # Lorenz63: 100, Lorenz96: 100
    lr = 0.05  # Lorez63: 0.05 , Lorenz96: 0.02
    optimizer = "adam"  # "adam, norm" Lorenz63:norm,  Lorenz96:adam,
    R_inv_type = 2  # ShallowWater 为0， 其他都是 2


args = Args()
FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))

# 为了画图用
folder_name = os.path.join(
    "./results/framework/", args.pde_problem, args.da_method)
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


if(args.pde_problem == "Lorenz96"):  # "Lorenz63,,Burgers,ShallowWater"
    print("-" * 20 + "Lorenz96 Problem" + "-" * 20)
    grid_info, state_info, time_info = construct_lorenz96_initial_state(
        F=8.0, sig_b=0.13, n=40)
    (state_init_pred_save, state_init_true) = state_info
    (time_vector, nt_time, ind_obs, nt_obs) = time_info
    Lorenz96_F = 8
    pde_model = Lorenz96(Lorenz96_F).to(device)
    R_sigma, B_sigma, R_mat, B_mat, R_inv_2, R_inv_1, R_inv_0, B_inv, state_background = construct_lorenz96_cov_matrix(
        state_init_true)


if(args.pde_problem == "Burgers"):  # "Lorenz63,,Burgers,ShallowWater"
    print("-"*20 + "Burgers Problem" + "-"*20)
    grid_info, state_info, time_info = construct_burgers_initial_state()
    (dx, x) = grid_info
    x = x.to(device)
    (state_init_pred_save, state_init_true) = state_info
    (time_vector, nt_time, ind_obs, nt_obs) = time_info

    pde_model = Burgers(x).to(device)

    R_sigma, B_sigma, R_mat, B_mat, R_inv_2, R_inv_1, R_inv_0, B_inv, state_background = construct_burgers_cov_matrix(
        state_init_true)


if(args.pde_problem == "ShallowWater"):  # "Lorenz63,,Burgers,ShallowWater"
    print("-"*20 + "Shallow water Problem" + "-"*20)
    grid_info, state_info, time_info = construct_shallowwater_initial_state()

    (dx, dy, grid_x, grid_y, vector_x, vector_y) = grid_info
    (state_init_pred_save, state_init_true) = state_info
    (time_vector, nt_time, ind_obs, nt_obs) = time_info

    pde_model = ShallowWaterModel(grid_info=(
        dx, dy, grid_x, grid_y, vector_x, vector_y)).to(device)

    # R_sigma, B_sigma, R_inv_0, B_inv_0, state_background = construct_shallowwater_cov_matrix(
    #     state_init_true)

    R_sigma, B_sigma, R_mat, B_mat, R_inv_2, R_inv_1, R_inv_0, B_inv, state_background = construct_shallowwater_cov_matrix_complex(
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


# %% 开始做4维同化


state_init_pred_save = state_background.to(device)
if(args.da_method in ("4dvar",  "auto", "unchainAuto")):
    B_mat = None
    B_inv = None


# 观测序列加扰动
state_full_obs = state_full_true[ind_obs] + \
    R_sigma * torch.randn(state_full_true[ind_obs].shape)
logger.info("-" * 20 + "start generate R,B matrix" + "-" * 20)


###########   begin assimilation  ###########
# classical 4dvar
if (args.da_method == "4dvar"):
    # learning rate 待调整, optimizer 待调整
    print("-"*20 + "classical 4dvar" + "-"*20)
    full_process_time_start = time.time()
    state_init_pred = state_init_pred_save.clone()
    state_init_pred.requires_grad = True

    with torch.no_grad():
        for iteration in range(args.total_iteration):
            print(f"iteration:{iteration}")
            time_start = time.time()
            state_final_predict, state_full_pred = da_odeint(
                pde_model, state_init_pred, time_vector, method=None, device=device)
            gradient = classical4dvar(state_full_pred=state_full_pred,
                                      state_full_obs=state_full_obs,
                                      func=pde_model,
                                      time_vector=time_vector,
                                      observe_ops=observe_ops,
                                      ind_m=ind_obs,
                                      R_inv=R_inv_2,
                                      R_inv_type=args.R_inv_type,
                                      state_background=state_background,
                                      state_init_pred=state_full_pred[0, :],
                                      B_inv=B_inv)

            print(f"grad:{gradient}")

            state_init_pred = state_init_pred - args.lr * \
                gradient / torch.linalg.norm(gradient)
            logger.info("elapse time:{}".format(time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))
    with torch.no_grad():
        state_final_pred, state_full_pred = da_odeint(
            pde_model, state_init_pred, time_vector, method=None)

# autogradient 4dvar
if (args.da_method == "auto"):
    print("-" * 20 + "autogradient 4dvar" + "-" * 20)
    full_process_time_start = time.time()

    state_init_pred = state_init_pred_save.clone()
    state_init_pred.requires_grad = True

    optimizer = Adam([state_init_pred], lr=args.lr)

    obs_len = len(ind_obs)

    for iteration in range(args.total_iteration):
        if(False):
            use_obs_len = np.random.randint(obs_len)
            use_obs_len = np.random.choice(np.arange(5, obs_len))
        else:
            use_obs_len = obs_len

        # print(use_obs_len)

        time_start = time.time()
        state_final_pred, state_full_pred = da_odeint(
            pde_model, state_init_pred, time_vector, method=None)
        loss = standard_da_loss(pred_input=observe_ops(state_full_pred[ind_obs[0:use_obs_len]]),
                                true_input=state_full_obs[0:use_obs_len],
                                R_inv=R_inv_2,
                                R_inv_type=args.R_inv_type,
                                state_background=state_background,
                                state_init_pred=state_full_pred[0, :],
                                B_inv=B_inv)

        if(args.optimizer == "norm"):
            loss.backward()
            print(f"grad:{state_init_pred.grad}")
            print(f"loss:{loss}")
            with torch.no_grad():
                gradient = torch.tensor(state_init_pred.grad)
                state_init_pred = state_init_pred - args.lr * \
                    gradient / torch.linalg.norm(gradient)
            state_init_pred.requires_grad = True
            print(state_init_pred)

        if(args.optimizer == "adam"):
            optimizer.zero_grad()
            loss.backward()
            print(f"grad:{state_init_pred.grad}")
            print(f"loss:{loss}")
            optimizer.step()

        logger.info("simgle step elapse time:{}".format(
            time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))
    with torch.no_grad():
        state_final_pred, state_full_pred = da_odeint(
            pde_model, state_init_pred, time_vector, method=None)


if (args.da_method == "unchainAuto"):
    print("-" * 20 + "unchained autogradient 4dvar by torchdiffq" + "-" * 20)
    full_process_time_start = time.time()

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    #############

    state_init_pred = state_init_pred_save.clone()
    state_init_pred.requires_grad = True

    optimizer = torch.optim.Adam([state_init_pred], lr=args.lr)

    for iteration in range(args.total_iteration):
        time_start = time.time()
        state_full_pred = odeint(pde_model, state_init_pred, time_vector,
                                 method=args.method).to(device)
        loss = standard_da_loss(pred_input=observe_ops(state_full_pred[ind_obs]),
                                true_input=state_full_obs,
                                R_inv=R_inv_2,
                                R_inv_type=args.R_inv_type,
                                state_background=state_background,
                                state_init_pred=state_full_pred[0, :],
                                B_inv=B_inv)

        if(args.optimizer == "norm"):
            loss.backward()
            print(f"grad:{state_init_pred.grad}")
            print(f"loss:{loss}")
            with torch.no_grad():
                gradient = torch.tensor(state_init_pred.grad)
                state_init_pred = state_init_pred - args.lr * \
                    gradient / torch.linalg.norm(gradient)
            state_init_pred.requires_grad = True
            print(state_init_pred)

        if(args.optimizer == "adam"):
            optimizer.zero_grad()
            loss.backward()
            print(f"grad:{state_init_pred.grad}")
            print(f"loss:{loss}")
            optimizer.step()

        with torch.no_grad():
            gradient = torch.tensor(state_init_pred.grad)
            state_init_pred = state_init_pred - args.lr * \
                gradient / torch.linalg.norm(gradient)
        state_init_pred.requires_grad = True

        logger.info("simgle step elapse time:{}".format(
            time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))

    with torch.no_grad():
        state_final_pred, state_full_pred = da_odeint(
            pde_model, state_init_pred, time_vector, method=None)


if (args.da_method == "KF"):
    print("-" * 20 + "kalman filter" + "-" * 20)

    full_process_time_start = time.time()

    km = 0
    B_matrix = B_mat
    state_full_pred = torch.empty(state_full_true.shape)
    state_full_pred[0, :] = state_background

    state_current = state_background
    for index, time_ in enumerate(time_vector[0:-1]):
        print(f"index:{index}")
        time_start = time.time()

        dt = time_vector[index + 1] - time_vector[index]
        # step forward 推出 index + 1的状态
        state_current = state_current + rk4_step(
            func=pde_model, t0=time_vector[index], dt=dt, y0=state_current)

        # 注意维度
        B_matrix = KF_cal_Bmatrix(B_matrix,
                                  pde_model, t0=time_vector[index], dt=dt,
                                  state_current=state_current)

        state_full_pred[index + 1] = state_current

        if ((km < nt_obs) and (index + 1) == ind_obs[km] and (True)):

            jacobian_h = jacobian(observe_ops, state_current)

            K_matrix = KF_cal_Kmatrix(
                state_current, jacobian_h, B_matrix, R_mat=R_mat)

            diff = state_full_obs[km, :] - observe_ops(state_current)

            state_current = KF_update_state(
                state_current, K_matrix, diff)

            state_full_pred[index + 1] = state_current

            P_matrix = KF_update_corvariance(K_matrix, jacobian_h, B_matrix)

            B_matrix = P_matrix
            km = km + 1

        logger.info("simgle step elapse time:{}".format(
            time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))


if (args.da_method == "enKF"):
    print("-" * 20 + "ensemble kalman filter" + "-" * 20)
    full_process_time_start = time.time()
    Ensenble_size = 20

    km = 0
    B_matrix = B_mat
    state_full_pred = torch.empty(state_full_true.shape)

    state_current = state_background
    state_len = len(state_current.shape)

    state_current_batch = torch.empty((Ensenble_size,) + state_current.shape)

    # print(B_matrix.shape[0:state_len], B_matrix.shape)
    # 转成square_matrix
    state_numel = state_current.shape.numel()
    B_matrix_square = B_matrix.reshape([state_numel, state_numel])
    corvariance_generator = MultivariateNormal(
        torch.zeros(B_matrix_square.shape[0]), B_matrix_square)

    R_mat_square = R_mat.reshape([state_numel, state_numel])
    R_corvariance_generator = MultivariateNormal(
        torch.zeros(R_mat_square.shape[0]), R_mat_square)

    for i in range(Ensenble_size):
        state_current_batch[i] = state_current + corvariance_generator.sample(
        ).reshape(state_current.shape)

    state_full_pred[0, :] = torch.mean(state_current_batch, dim=0)

    for index, time_ in enumerate(time_vector[0:-1]):
        time_start = time.time()
        dt = time_vector[index + 1] - time_vector[index]

        for i in range(Ensenble_size):
            state_current_batch[i] = state_current_batch[i] + rk4_step(
                func=pde_model, t0=time_vector[index], dt=dt, y0=state_current_batch[i])

        state_current_batch_mean = torch.mean(state_current_batch, dim=0)
        state_full_pred[index +
                        1], B_matrix = KF_update_state_corvariance_ensemble(state_current_batch)

        if ((km < nt_obs) and (index + 1) == ind_obs[km] and (True)):

            jacobian_h = jacobian(observe_ops, state_current)

            K_matrix = KF_cal_Kmatrix(
                state_current_batch_mean, jacobian_h, B_matrix, R_mat)

            diff_batch = torch.empty(state_current_batch.shape)
            for i in range(Ensenble_size):
                # create virtual observations
                obs_fluctuation = state_full_obs[km,
                                                 :] + R_corvariance_generator.sample().reshape(state_current.shape)
                diff = obs_fluctuation - observe_ops(state_current_batch[i])
                diff_batch[i, :] = diff

            # uai[:, i] = ubi[:, i] + K @ (wi[:, i]-ObsOp(ubi[:, i]))
            state_current_batch = KF_update_state_batch(
                state_current_batch, K_matrix, diff_batch)

            # 更新分析场, B矩阵
            state_full_pred[index +
                            1], P_matrix = KF_update_state_corvariance_ensemble(state_current_batch)

            B_matrix = P_matrix
            km = km + 1
        logger.info("simgle step elapse time:{}".format(
            time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))


if (args.da_method == "3dvar"):
    print("-" * 20 + "3dvar" + "-" * 20)
    full_process_time_start = time.time()

    km = 0
    B_matrix = 5.0*R_mat
    # B_mat = R_mat
    state_full_pred = torch.empty(state_full_true.shape)
    state_full_pred[0, :] = state_background

    state_current = state_background
    for index, time_ in enumerate(time_vector[0:-1]):
        print(f"index:{index}")
        time_start = time.time()
        dt = time_vector[index + 1] - time_vector[index]
        # step forward 推出 index + 1的状态
        state_current = state_current + rk4_step(
            func=pde_model, t0=time_vector[index], dt=dt, y0=state_current)

        # 注意维度
        state_full_pred[index + 1] = state_current

        if ((km < nt_obs) and (index + 1) == ind_obs[km]):

            state_current = classical3dvar(
                observe_ops, state_current, state_full_obs[km, :], R_mat,  B_matrix)

            state_full_pred[index + 1] = state_current
            km = km + 1

        logger.info("simgle step elapse time:{}".format(
            time.time() - time_start))

    logger.info("full process elapse time:{}".format(
        time.time() - full_process_time_start))


# %% 后验测试同化效果
if (args.pde_problem == "Lorenz63"):
    file_name = args.prefix + "_result_4dvar.png"
    folder_name = os.path.join(
        "./results/framework/", args.pde_problem, args.da_method)
    plot_Lorenz63_onevector(
        time_vector, state_full_true,
        time_vector[ind_obs], state_full_obs,
        time_vector, state_full_pred, variable_index=0, fig_index=0,
        foldername=folder_name, filename=file_name)

    mean_error = evaluate_lorenz63_laststate(
        time_vector, state_full_true, time_vector, state_full_pred, state_number=ind_obs[-1])
    logger.info(f"mean_error:{mean_error}")

    filename = args.prefix + "_errormap_4dvar.png"
    plot_Lorenz63_errormap(
        state_full_true, state_full_pred, foldername=folder_name, filename=filename)


if (args.pde_problem == "Lorenz96"):
    file_name = args.prefix + "_result.png"
    folder_name = os.path.join(
        "./results/framework/", args.pde_problem, args.da_method)
    plot_Lorenz96_onevector(
        time_vector, state_full_true,
        time_vector[ind_obs], state_full_obs,
        time_vector, state_full_pred, variable_index=0, fig_index=0,
        foldername=folder_name, filename=file_name)

    mean_error = evaluate_lorenz96_laststate(
        time_vector, state_full_true, time_vector, state_full_pred, state_number=ind_obs[-1])
    logger.info(f"mean_error:{mean_error}")

    filename = args.prefix + "_errormap.png"
    plot_Lorenz96_errormap(
        state_full_true, state_full_pred, foldername=folder_name, filename=filename)


if (args.pde_problem == "Burgers"):
    file_name = args.prefix + "_result.png"
    folder_name = os.path.join(
        "./results/framework/", args.pde_problem, args.da_method)

    mean_error = evaluate_lorenz63_laststate(
        time_vector, state_full_true, time_vector, state_full_pred, state_number=ind_obs[-1])
    logger.info(f"mean_error:{mean_error}")

    plot_Burgers_onevector(
        time_vector, state_full_true,
        time_vector[ind_obs], state_full_obs,
        time_vector, state_full_pred, variable_index=0, fig_index=0,
        foldername=folder_name, filename=file_name, x=x)

    filename = args.prefix + "_errormap.png"
    plot_Burgers_errormap(
        state_full_true, state_full_pred, foldername=folder_name, filename=filename)


if (args.pde_problem == "ShallowWater"):
    mean_error = evaluate_lorenz63_laststate(
        time_vector, state_full_true[:, 0, :], time_vector, state_full_pred[:, 0, :], state_number=ind_obs[-1])

    logger.info(f"mean_error:{mean_error}")
    filename = args.prefix + "_errormap.png"
    plot_ShallowWater_laststate(
        state_full_true, state_full_pred, foldername=folder_name, filename=filename)
