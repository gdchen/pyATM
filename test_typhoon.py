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

from dynamic_model.Typhoon_sync import Typhoon_sync, construct_typhoonsync_initial_state


# 为了画图用
foldername = os.path.join(
    "./results/typhoon/")
FileUtils.makedir(foldername)


### read basic info  ###
grid_info, (state0_init,
            state0_true), time_info = construct_typhoonsync_initial_state()
dx, dy, grid_x, grid_y, vector_x, vector_y = grid_info
(time_vector, nt_time, ind_obs, nt_obs) = time_info

print(state0_true.shape)

# filename = "result.png"
pde_model = Typhoon_sync(grid_info)


with torch.no_grad():
    plot_2d(pde_model.calculate_omega_from_uv(state0_true[2, :, :].unsqueeze(0),
                                              state0_true[3, :, :].unsqueeze(0)).squeeze(), foldername, "result_omg.png")

    plot_2d(state0_true[1, :, :], foldername, "result_phi.png")

    plot_2d(state0_true[2, :, :], foldername, "result_u.png")

    plot_2d(state0_true[3, :, :], foldername, "result_v.png")

    plot_2d(state0_true[4, :, :], foldername, "result_p.png")
    plot_2d(state0_true[5, :, :], foldername, "result_c.png")


device = "cpu"

# original_state.requires_grad = True

# state = original_state.clone()

original_state = state0_true
state = original_state.clone()

obs_true = torch.empty(size=tuple(
    (len(ind_obs),)) + tuple(state.shape)).to(device)
obs_simu = torch.empty(size=tuple(
    (len(ind_obs),)) + tuple(state.shape)).to(device)


obs_count = 0
for index, time in enumerate(time_vector[1::]):
    dt = time_vector[index + 1] - time_vector[index]
    state_star = state + \
        rk4_step(pde_model, time, dt,  state, device=device)
    _, _, u, v, pressure, c = state_star
    u = u.unsqueeze(0)
    v = v.unsqueeze(0)
    c = c.unsqueeze(0)
    pressure = pressure.unsqueeze(0)
    dp = pde_model.get_pressuere(u, v)
    u = u - pde_model.grad_x(dp)
    v = v - pde_model.grad_y(dp)
    pressure = pressure + dp
    state = torch.cat([u, u, u, v, pressure, c], dim=0)
    if(index in ind_obs):
        obs_true[obs_count, :] = state
        obs_count += 1

print(obs_true.shape)


def simulation(init_state, time_vector, ind_obs):
    state = init_state
    obs_count = 0
    for index, time in enumerate(time_vector[1::]):
        dt = time_vector[index + 1] - time_vector[index]
        state_star = state + \
            rk4_step(pde_model, time, dt,  state, device=device)
        _, _, u_star, v_star, pressure_star, c_star = state_star
        u_star, v_star, c_star, pressure_star = u_star.unsqueeze(
            0), v_star.unsqueeze(0), c_star.unsqueeze(0), pressure_star.unsqueeze(0)

        dp = pde_model.get_pressuere(u_star, v_star)
        u = u_star - pde_model.grad_x(dp)
        v = v_star - pde_model.grad_y(dp)
        pressure = pressure_star + dp
        state = torch.cat([torch.zeros(u.shape), torch.zeros(
            u.shape), u, v, pressure, c_star], dim=0)
        if(index in ind_obs):
            obs_simu[obs_count, :] = state
            obs_count += 1
    return obs_simu


obs_true_detach = obs_true.detach()
obs_true_detach.requires_grad = False

criterion = nn.MSELoss()

# torch.autograd.set_detect_anomaly(True)
state_init = state0_init.clone()
state_init.requires_grad = True
optimizer = torch.optim.Adam([state_init], lr=0.02)


print("#" * 20)
print("start data_ assimilation")
for iteration in range(100):
    # state_init = state_init.detach()
    obs_simu = simulation(state_init, time_vector, ind_obs)

    # obs_simu = torch.tile(state_init, [10, 1, 1, 1])
    loss = criterion(obs_simu[:, 2:3, :, :], obs_true_detach[:, 2:3, :, :])
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    obs_simu = obs_simu.detach()

    print("#" * 30)
    print(iteration)
    print(loss)
    state_init = state_init.detach()
    state_init.requires_grad = True
    optimizer = torch.optim.Adam([state_init], lr=0.02)

    omega, phi, u, v, p, c = state_init
    omega = pde_model.calculate_omega_from_uv(u.unsqueeze(0), v.unsqueeze(0))

    plot_2d(u.detach(), foldername, "result_u_da" +
            str(iteration).zfill(3) + ".png")
    plot_2d(v.detach(), foldername, "result_v_da" +
            str(iteration).zfill(3) + "png")
    plot_2d(p.detach(), foldername, "result_p_da" +
            str(iteration).zfill(3) + "png")
    plot_2d(omega.squeeze().detach(), foldername,
            "result_omg_da" + str(iteration).zfill(3) + "png")
    plot_2d(c.squeeze().detach(), foldername,
            "result_c_da" + str(iteration).zfill(3) + "png")

# loss = torch.sum(torch.abs(state))
# loss.backward()
# print("grad")
# print(original_state.grad)
# print(p.shape)
# plot_2d(p.detach(), foldername, "result_p_temp.png")
# exit()


# state_final_true, state_full_true = da_odeint(
#     pde_model, state0_true, time_vector, method=None, device=device)
