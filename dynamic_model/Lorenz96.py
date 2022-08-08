#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:03:27 2021

@author: yaoyichen
"""

import torch
import torch.nn as nn
import numpy as np
import os


class Lorenz96(nn.Module):
    """
    # Lorenz 96 model
    """

    def __init__(self, *args):
        super(Lorenz96, self).__init__()
        self.F = args[0]

    def forward(self, t, state):
        n = len(state)
        f = torch.zeros(state.shape)

        state_p1 = torch.roll(state, -1, 0)
        state_m2 = torch.roll(state, 2, 0)
        state_m1 = torch.roll(state, 1, 0)

        f = (state_p1 - state_m2)*state_m1 - state

        # f[0] = (state[1] - state[n-2]) * state[n-1] - state[0]
        # f[1] = (state[2] - state[n-1]) * state[0] - state[1]
        # f[n-1] = (state[0] - state[n-3]) * state[n-2] - state[n-1]
        # for i in range(2, n-1):
        #     f[i] = (state[i+1] - state[i-2]) * state[i-1] - state[i]
        # Add the forcing term
        f = f + self.F
        return f

    def observe(self, u):
        return u


def construct_lorenz96_initial_state(F, sig_b, n):
    #### grid info ###
    grid_info = ()

    #### state info ####
    u0_true = F * torch.ones(n)  # Init
    u0_true[0] = u0_true[0] + 0.2
    u0_true[3] = u0_true[0] + 0.2
    u0_true[10] = u0_true[10] + 0.25
    u0_true[20] = u0_true[20] + 0.20
    u0_true[30] = u0_true[30] + 0.1
    u0_true[35] = u0_true[35] - 0.3
    u = u0_true + torch.randn([n, ]) * sig_b

    state_info = (u, u0_true)

    #### time info  ####
    nt_time = 300
    time_vector = torch.linspace(0., 0.01 * nt_time, nt_time + 1)
    ind_obs = torch.arange(20, 120, 20)
    nt_obs = len(ind_obs)

    time_info = (time_vector, nt_time, ind_obs, nt_obs)

    return grid_info, state_info, time_info


def construct_lorenz96_cov_matrix(state_init_true):
    n = state_init_true.shape[0]

    R_sigma = 0.3
    B_sigma = 0.13
    sig_b = B_sigma
    state_background = state_init_true + torch.randn([n, ]) * sig_b

    R_base = R_sigma*torch.eye(state_init_true.shape.numel())
    R_mat = R_base.reshape(state_init_true.shape + state_init_true.shape)
    R_inv_2 = torch.linalg.inv(R_base).reshape(
        state_init_true.shape + state_init_true.shape)
    R_inv_1 = torch.diagonal(torch.linalg.inv(
        R_base), 0).reshape(state_init_true.shape)
    R_inv_0 = torch.tensor([1.0/R_sigma])

    if(True):
        # state_background = state_init_true + \
        #     B_sigma * torch.randn(state_init_true.shape)
        B_base = B_sigma*torch.eye(state_init_true.shape.numel())
        B_mat = B_base.reshape(state_init_true.shape + state_init_true.shape)
        B_inv = torch.linalg.inv(B_base).reshape(
            state_init_true.shape + state_init_true.shape)
    return R_sigma, B_sigma, R_mat, B_mat, R_inv_2, R_inv_1, R_inv_0, B_inv, state_background


def plot_Lorenz96(time, state, time_obs, obs, fig_index):
    """
    plot the basic figure of the Lorenz63 model
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(fig_index, figsize=(6, 4))
    plt.plot(time, state[0, :], c="tab:blue")
    plt.plot(time, state[1, :], c="tab:orange")
    plt.plot(time, state[2, :], c="tab:green")
    plt.plot(time_obs, obs[0, :], 'x', c="tab:blue")
    plt.plot(time_obs, obs[1, :], 'x', c="tab:orange")
    plt.plot(time_obs, obs[2, :], 'x', c="tab:green")
    plt.xlabel("time")
    plt.ylabel("state value")


def plot_Lorenz96_onevector(time_true, state_true, time_obs, state_obs,
                            time_analysis, state_analysis, variable_index, fig_index,
                            foldername, filename):
    import matplotlib.pyplot as plt
    plt.figure(fig_index)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    k = 0
    ax.plot(time_true, state_true[:, variable_index],
            label=r'true', linewidth=2, color="k")
    ax.plot(time_obs, state_obs[:, variable_index], 'x', fillstyle='none',
            label=r'observation', markersize=5, markeredgewidth=2, color="red")
    ax.plot(time_analysis, state_analysis[:, variable_index],
            '--', label=r'analysis', linewidth=2)
    ax.set_xlabel(r'time', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(foldername, filename),
                dpi=500, bbox_inches='tight')


def plot_Lorenz96_errormap(state_true, state_analysis, foldername, filename):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    error = torch.abs(state_true - state_analysis)

    fig, ax = plt.subplots(3, 1)

    pcm = ax[0].contourf(torch.t(state_true).numpy(),
                         cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax[0], extend='max')

    pcm = ax[1].contourf(torch.t(state_analysis).numpy(),
                         cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax[1], extend='max')

    pcm = ax[2].contourf(torch.t(error).numpy(), cmap='PuBu_r', shading='auto')

    fig.colorbar(pcm, ax=ax[2], extend='max')

    plt.savefig(os.path.join(foldername, filename),
                dpi=500, bbox_inches='tight')


def evaluate_lorenz96_laststate(time_true, state_true, time_analysis, state_analysis, state_number):
    if(state_true.shape != state_analysis.shape):
        print("the shape of the result is not right")

    last_true = state_true[-state_number::, :]
    last_analysis = state_analysis[-state_number::, :]

    error_mean = torch.mean(torch.abs(last_true - last_analysis))
    print(f"error:{error_mean}")
