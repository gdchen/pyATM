#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""
# %%
import os
import torch
import torch.nn as nn
import time
import numpy as np
from neural_model.differ_module import Order2_Diff1_Unstructure, Order2_Diff1, Order2_Diff2_Unstructure, Order2_Diff2
from neural_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period, Order2_Diff2_Unstructure, Order2_Diff1_Unstructure
import matplotlib.pyplot as plt


class Args:
    method = "rk4"
    adjoint = False


args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class Boundary_Model(nn.Module):
    def __init__(self, grid_info, simu_para):
        super(Boundary_Model, self).__init__()

        self.x, self.y = grid_info
        self.nx = len(x)
        self.ny = len(y)
        self.mu, self.Ue = simu_para
        self.Ue = self.Ue.unsqueeze(0)
        self.diff1_x = Order2_Diff1_Unstructure(
            self.x, total_dim=2, diff_dim=1)
        self.diff1_x_dim1 = Order2_Diff1_Unstructure(
            self.x, total_dim=1, diff_dim=1)

        self.diff2_x = Order2_Diff2_Unstructure(
            self.x, total_dim=2, diff_dim=1)

        self.diff1_y = Order2_Diff1_Unstructure(
            self.y, total_dim=2, diff_dim=2)
        self.diff2_y = Order2_Diff2_Unstructure(
            self.y, total_dim=2, diff_dim=2)

        self.dUe = self.diff1_x_dim1(self.Ue)*self.Ue

    def forward(self, t, state):
        """
        shape [6,10,10,20]
        """
        # u, v = state
        u, v = state
        u, v = u.unsqueeze(0), v.unsqueeze(0)

        du = self.mu * (self.diff2_y(u) + self.diff2_x(u)) - \
            u * self.diff1_x(u) - v * self.diff1_y(u) + 0.01
        dv = self.mu * (self.diff2_y(v) + self.diff2_x(v)) - u * \
            self.diff1_x(v) - v * self.diff1_y(v)

        # 外部导数为0
        du[0, :, self.ny - 1] = 0

        # 壁面导数为0
        dv[0, :, 0] = 0
        du[0, :, 0] = 0

        # 入口导数为0
        du[0, 0, :] = 0
        dv[0, 0, :] = 0

        du, dv = du.squeeze(), dv.squeeze()
        result = torch.stack([du, dv])
        return result


def construct_initial_state():
    """
    construct the initial problem definition
    """
    nx, ny = 100,  100
    dx = 0.02
    dy = 0.01
    mu = 1e-4
    x = torch.arange(0, nx * dx, dx, dtype=torch.float32)
    y_linear = torch.arange(0, ny * dy, dy, dtype=torch.float32)
    y_tanh = torch.tanh(2 * y_linear)
    y_tanh_scale = (y_tanh - y_tanh[0]) / (y_tanh[-1] - y_tanh[0])
    y = 1 - y_tanh_scale.flip(dims=(0,))

    # y = y_linear
    Ue = 1.0 - 0.0 * x/nx * dx
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_info = (x, y)

    u0_true = torch.ones(nx, 1) * 1.0 * (1 - (1 - y) ** 2)
    v0_true = 1e-4*torch.ones(nx, ny)

    state = torch.stack(
        [u0_true, v0_true])
    simu_para = mu, Ue
    return grid_info, simu_para, state


grid_info, simu_para, state_True = construct_initial_state()


(x, y) = grid_info
print(x, y)
device = "cpu"
# print(result)
pde_model = Boundary_Model(grid_info, simu_para).to(device)

t = torch.linspace(0., 2.0*3, 100*3).to(device)
# %%
start_time = time.time()
y_true = odeint(pde_model, state_True, t, method=args.method).to(device)

print(y_true.shape)
print(y_true[-1, ...].mean(dim=(0, 1)))


fig, ax = fig, ax = plt.subplots(figsize=(10, 4))
plt.plot(y_true[90, 0, 0, :].squeeze().cpu().detach().numpy(),
         y, '-x', markersize=2, label="0")
plt.plot(y_true[90, 0, 50, :].squeeze().cpu().detach().numpy(),
         y, '-x', markersize=2, label="100")
plt.plot(y_true[90, 0, 99, :].squeeze().cpu().detach().numpy(),
         y, '-x', markersize=2, label="900")

plt.legend()
plt.savefig("boundary.png")
