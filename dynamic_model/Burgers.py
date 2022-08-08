#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:03:27 2021

@author: yaoyichen
"""
import os
import torch
import torch.nn as nn
import numpy as np
from .differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period


class Burgers(nn.Module):
    def __init__(self, x_vector):
        super(Burgers, self).__init__()
        self.term1 = Order2_Diff1_Unstructure_Period(
            x_vector, total_dim=1, diff_dim=1)
        self.term2 = Order2_Diff2_Unstructure_Period(
            x_vector, total_dim=1, diff_dim=1)

    def forward(self, t, u):
        result = (- (u + 0.5) * self.term1.forward(u) +
                  0.0001 * self.term2.forward(u))
        return result

    def observe(self, u):
        return u


def construct_burgers_initial_state():
    """
    construct the initial problem definition
    better not include any randomness
    """
    #### grid info ####
    dx = 0.01
    x = torch.arange(0, 100 * dx, dx, dtype=torch.float32)

    grid_info = (dx, x)

    #### state info ####
    u = 1.0 + torch.sin(2 * np.pi * x).to(torch.float32)

    u0_true = 1.0 \
        + 0.2 * torch.sin(2*np.pi * x) \
        + 0.1*torch.cos(6*np.pi*x + 1.0/3.0) \
        + 0.1*torch.sin(10*np.pi*x + 5.0/9.0)

    u = u.unsqueeze(0)
    u0_true = u0_true.unsqueeze(0)

    state_info = (u, u0_true)

    #### time info  ####

    nt_time = 500
    time_vector = torch.linspace(0., 0.003 * nt_time, nt_time + 1)

    ind_obs = torch.arange(20, 220, 20)
    nt_obs = len(ind_obs)

    time_info = (time_vector, nt_time, ind_obs, nt_obs)

    return grid_info, state_info, time_info


def construct_burgers_cov_matrix(state_init_true):
    n = state_init_true.shape[0]

    R_sigma = 0.1
    B_sigma = 0.1
    sig_b = B_sigma

    dx = 0.01
    x = torch.arange(0, 100 * dx, dx, dtype=torch.float32)
    state_background = 1.0 + \
        torch.sin(2 * np.pi * x).unsqueeze(0).to(torch.float32)

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


# def plot():
#     plt.plot(state_full_true[0, :].squeeze().cpu().detach().numpy())
#     plt.plot(state_full_true[100, :].squeeze().cpu().detach().numpy())
#     plt.plot(state_full_true[200, :].squeeze().cpu().detach().numpy())
#     plt.plot(state_full_true[-1, :].squeeze().cpu().detach().numpy())

# (dx, x), state_init_pred, state_init_true = construct_initial_state()


# pde model
# pde_model = Burgers(x)
# print(pde_model)
# print(f"model parameter:{ModelUtils.get_parameter_number(pde_model)}")


def plot_Burgers_onevector(time_true, state_true, time_obs, state_obs,
                           time_analysis, state_analysis, variable_index, fig_index,
                           foldername, filename, x):  # %%

    import matplotlib.pyplot as plt
    print("=> ploting_result")
    with torch.no_grad():
        plt.figure(0)
        plt.clf()
        plt.title("")
        # plt.plot(x.cpu().detach().numpy(), state_true[0, :].cpu().squeeze(
        # ).detach().numpy(), '-k', label="true_start")
        plt.plot(x.cpu().detach().numpy(), state_true[-1, :].cpu().squeeze().detach(
        ).numpy(), label="true_end")

        # plt.plot(x.cpu().detach().numpy(), state_analysis[0, :].cpu().squeeze(
        # ).detach().numpy(), 'tab:gray', label="pred_start")
        plt.plot(x.cpu().detach().numpy(), state_analysis[-1, :].cpu().squeeze().detach(
        ).numpy(), label="assimilation_end")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(foldername, filename),
                    dpi=500, bbox_inches='tight')


def plot_Burgers_errormap(state_true, state_analysis, foldername, filename):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    error = torch.abs(state_true - state_analysis)

    fig, ax = plt.subplots(3, 1)

    vmin = torch.min(state_true)
    vmax = torch.max(state_true)

    pcm = ax[0].contourf(torch.t(state_true.squeeze()).numpy(),
                         cmap='PuBu_r', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax[0], extend='max')

    pcm = ax[1].contourf(torch.t(state_analysis.squeeze()).numpy(),
                         cmap='PuBu_r', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax[1], extend='max')

    pcm = ax[2].contourf(torch.t(error.squeeze()).numpy(),
                         cmap='PuBu_r', shading='auto')
    fig.colorbar(pcm, ax=ax[2], extend='max')

    plt.savefig(os.path.join(foldername, filename),
                dpi=500, bbox_inches='tight')
