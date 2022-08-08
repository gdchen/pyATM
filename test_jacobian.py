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
import torch.nn as nn
import logging
import time


import matplotlib.pyplot as plt

from neural_model.integral_module import RK4Method, JRK4Method
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.variational_tools import classical4dvar, standard_da_loss


class Args:
    main_folder = "framework"   # 问题定义
    sub_folder = "L63"  # 主方案定义
    prefix = "4dvar"


args = Args()


FileUtils.makedir(os.path.join("runs", args.main_folder, args.sub_folder))
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
logger.info(f"device:{device}")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True


class Lorenze63_H(nn.Module):
    def __init__(self):
        super(Lorenze63_H, self).__init__()
        pass

    def forward(self, state):
        return state


class Lorenz63(nn.Module):
    """
    # Lorenz 63 model
    """

    def __init__(self, *args):
        super(Lorenz63, self).__init__()
        self.sigma = args[0]
        self.beta = args[1]
        self.rho = args[2]

    def forward(self, t, state):
        # Unpack the state vector
        x, y, z = state[0], state[1], state[2]
        f = torch.zeros(state.shape)  # Derivatives
        f[0] = self.sigma * (y - x)
        f[1] = x * (self.rho - z) - y
        f[2] = x * y - self.beta * z
        return f


def plot_Lorenz63(time, state, fig_index):
    """
    plot the basic figure of the Lorenz63 model
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(fig_index, figsize=(6, 4))
    plt.plot(time, state[:, 0], c="tab:blue")
    plt.plot(time, state[:, 1], c="tab:orange")
    plt.plot(time, state[:, 2], c="tab:green")
    plt.xlabel("time")
    plt.ylabel("state value")


class Pipeline_model(nn.Module):
    def __init__(self, time_method, time_vector,
                 obs_map={0: 100, 1: 200, 2: 300}
                 ):
        super(Pipeline_model, self).__init__()
        self.time_method = time_method
        self.time_vector = time_vector
        self.obs_map = obs_map

    def forward(self, state):
        size = tuple(
            self.time_vector.shape) + tuple(state.shape)
        full_state = torch.empty(size=tuple(
            self.time_vector.shape) + tuple(state.shape))
        full_state[0, :] = state
        for index, time in enumerate(self.time_vector[1::]):
            state = self.time_method(time, state)
            full_state[index+1, :] = state
        return state, full_state


# pde model
sigma = 10.0
beta = 8.0/3.0
rho = 28.0
pde_model = Lorenz63(sigma, beta, rho)
logger.info(f"model parameter:{ModelUtils.get_parameter_number(pde_model)}")

# time forward
N = 40
time_vector = torch.linspace(0., 0.02*N, N+1).to(device)
time_forward_method = RK4Method(pde_model, dt=0.02).to(device)
inverse_time_forward_method = JRK4Method(pde_model, dt=0.02).to(device)

# twin framework
model_true = Pipeline_model(time_forward_method,
                            time_vector=time_vector).to(device)
observe_ops = Lorenze63_H()

with torch.no_grad():
    state_init_true = torch.tensor([1.0, 1.0, 1.0]).to(device)
    state_final_true, state_full_true = model_true(state_init_true)
    print(state_final_true.shape, state_full_true.shape)
    plot_Lorenz63(time_vector.cpu().detach().numpy(),
                  state_full_true.cpu().detach().numpy(), fig_index=0)


# %% 开始做4维同化
state_init_pred = torch.tensor([1.01, 1.01, 1.01]).to(device)
ind_m = torch.arange(7, 40, 2)
R = torch.eye(3)
R_inv = torch.linalg.inv(R)

# 4dvar
if (True):
    print("-"*20 + "classical 4dvar" + "-"*20)
    total_iteration = 1
    for i in range(total_iteration):

        time_start = time.time()
        state_final_pred, state_full_pred = model_true(state_init_pred)
        dJ = classical4dvar(state_full_pred, state_full_true,
                            model_true, inverse_time_forward_method, observe_ops, ind_m, R_inv)
        print(f"grad:{dJ}")
        state_init_pred = state_init_pred - dJ * 0.01
        print(state_init_pred)
        logger.info("elapse time:{}".format(time.time() - time_start))


# autogradient
if (True):
    print("-"*20 + "autogradient 4dvar" + "-"*20)
    state_init_pred = torch.tensor([1.01, 1.01, 1.01]).to(device)
    state_init_pred.requires_grad = True
    optimizer = torch.optim.Adam([state_init_pred], lr=0.01)
    criterion = torch.nn.MSELoss()

    total_iteration = 1  # 2000
    for iteration in range(total_iteration):
        time_start = time.time()
        state_final_pred, state_full_pred = model_true(state_init_pred)

        loss = standard_da_loss(pred_input=observe_ops(state_full_pred[ind_m]),
                                true_input=observe_ops(state_full_true[ind_m]),
                                R_inv=R_inv)

        diff = observe_ops(state_full_pred[ind_m]) - observe_ops(
            state_full_true[ind_m])

        print(loss)
        optimizer.zero_grad()
        loss.backward()
        print(f"grad:{state_init_pred.grad}")
        optimizer.step()
        logger.info("elapse time:{}".format(time.time() - time_start))
    #

    # %%
