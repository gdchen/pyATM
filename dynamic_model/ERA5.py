#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:22:08 2021

@author: yaoyichen
"""

import torch
import torch.nn as nn
import numpy as np
import os

from dynamic_model.differ_module import Order2_Diff1_Unstructure, Order2_Diff2_Unstructure


def getUV_height5(foldername, datestr):
    """
    return 24, 2, 157, 132
    """
    target_heihgt = 0
    filename = datestr + "0".zfill(2) + "00.npy"
    data_org = np.load(os.path.join(foldername, filename))
    Nx, Ny = data_org.shape[0:2]

    result = np.empty([24, 2, Nx, Ny])
    for i in range(24):
        filename = datestr + str(i).zfill(2) + "00.npy"
        print(filename)
        data_org = np.load(os.path.join(foldername, filename))
        data_reshape = data_org.reshape((Nx, Ny, 2, 10))
        data_target = np.squeeze(data_reshape[:, :, :, target_heihgt])
        # 换算到km/h
        result[i, :] = np.einsum("xyv->vxy", data_target)*3.6/10.0

    print(result.shape)
    print(np.expand_dims(result[:, 1, :, :], 1).shape)
    result = np.concatenate((np.expand_dims(
        result[:, 1, :, :], 1), np.expand_dims(result[:, 0, :, :], 1)), axis=1)
    return result


class ERA5_basic(nn.Module):
    """
    # ERA5_basic model
    """

    def __init__(self, grid_info):
        super(ERA5_basic, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y = grid_info

        self.mu = 0.001
        self.diff1_x = Order2_Diff1_Unstructure(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff1_y = Order2_Diff1_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

        self.diff2_x = Order2_Diff2_Unstructure(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

    def forward(self, t, state):
        # print("#" * 20)
        # print(state.shape)
        u, v = state
        u, v = u.unsqueeze(0), v.unsqueeze(0)
        du = -u * self.diff1_x(u) - v * self.diff1_y(u) + \
            self.mu*(self.diff2_x(u) + self.diff2_y(u))
        dv = -u * self.diff1_x(v) - v * self.diff1_y(v) + \
            self.mu*(self.diff2_x(v) + self.diff2_y(v))

        result = torch.stack([du.squeeze(), dv.squeeze()])
        result[:, 0, :] = 0.0
        result[:, -1, :] = 0.0
        result[:, :, 0] = 0.0
        result[:, :, -1] = 0.0

        return result

    def observe(self, u):
        return u

    def calcualte_divergence(self, state):

        u, v = state
        u, v = u.unsqueeze(0), v.unsqueeze(0)
        divergence = self.diff1_x(u) + self.diff1_y(v)
        return divergence


def construct_ERA5_initial_state():
    """
    construct the initial problem definition
    dx: 25km
    dy: 25km
    """

    foldername = "/Users/yaoyichen/project_earth/dataset/era5_one_day/uv"
    datestr = "20190101"
    result = getUV_height5(foldername, datestr)

    dx, dy = 25, 25
    Nx, Ny = result.shape[2:4]

    vector_x = torch.arange(0, Nx * dx, dx, dtype=torch.float32)
    vector_y = torch.arange(0, Ny * dy, dy, dtype=torch.float32)

    # 结构化和非结构化网格的切换
    # x = 0.5+0.5*(torch.tanh(2.0*(x_temp-0.5)) /
    #              torch.tanh(2.0 * (torch.tensor(1.0) - 0.5)))
    grid_x, grid_y = torch.meshgrid(vector_x, vector_y)
    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y)

    state_info = torch.tensor(np.squeeze(result[0, :]), dtype=torch.float32)
    full_state_info = torch.tensor(result,  dtype=torch.float32)

    #### time info  ####
    delta_hour = 0.1
    nt_time = int(23/delta_hour)
    time_vector = torch.linspace(0., delta_hour * nt_time, nt_time + 1)

    ind_obs = torch.arange(0, int(23/delta_hour) + 1, int(1/delta_hour))
    nt_obs = len(ind_obs)

    time_info = (time_vector, nt_time, ind_obs, nt_obs)

    return grid_info, state_info, time_info, full_state_info


def plot_ERA5(input_1, input_2, foldername, filename):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    fig, ax = plt.subplots(2, 1)

    vmax, vmin = 8, -8
    pcm = ax[0].contourf(input_1.squeeze().numpy(),
                         cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax[0], extend='max')
    ax[0].set_aspect("equal")

    pcm = ax[1].contourf(input_2.squeeze().numpy(),
                         cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax[1], extend='max')
    ax[1].set_aspect("equal")

    plt.savefig(os.path.join(foldername, filename),
                dpi=500, bbox_inches='tight')
