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


class Rayleigh_Bernard_Model(nn.Module):
    def __init__(self, grid_info, simu_para):
        super(Rayleigh_Bernard_Model, self).__init__()

        x, y, z = grid_info

        self.cp, self.cv, self.mu, self.k, self.g = simu_para
        self.diff1_x = Order2_Diff1_Unstructure_Period(
            x, total_dim=3, diff_dim=1)
        self.diff1_y = Order2_Diff1_Unstructure_Period(
            y, total_dim=3, diff_dim=2)
        self.diff1_z = Order2_Diff1_Unstructure(z, total_dim=3, diff_dim=3)

        self.diff2_x = Order2_Diff2_Unstructure_Period(
            x, total_dim=3, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure_Period(
            y, total_dim=3, diff_dim=2)
        self.diff2_z = Order2_Diff2_Unstructure(z, total_dim=3, diff_dim=3)

    def get_p(self, rou, T):
        return (self.cp - self.cv)*rou*T

    def forward(self, t, state):
        """
        shape [6,10,10,20]
        """
        u, v, w, rou, T, c = state
        u, v, w, rou, T, c = u.unsqueeze(0), v.unsqueeze(
            0), w.unsqueeze(0), rou.unsqueeze(0), T.unsqueeze(0), c.unsqueeze(0)
        p = self.get_p(rou, T)
        drou = - self.diff1_x(rou * u)
        - self.diff1_y(rou * v) \
            - self.diff1_z(rou * w)

        du = (- self.diff1_x(p) + self.mu *
              (self.diff2_x(u) + self.diff2_y(u) + self.diff2_z(u))) / rou \
            - u * self.diff1_x(u) - v * self.diff1_y(u) - w * self.diff1_z(u)

        dv = (- self.diff1_y(p) + self.mu * (self.diff2_x(v) + self.diff2_y(v) + self.diff2_z(v)))/rou \
            - u * self.diff1_x(v) - v * self.diff1_y(v) - w * self.diff1_z(v)

        dw = (-self.g * rou - self.diff1_z(p) + self.mu * (self.diff2_x(w) + self.diff2_y(w) + self.diff2_z(w))) / rou \
            - u * self.diff1_x(w) - v * self.diff1_y(w) - w * self.diff1_z(w)

        dT = (self.k * (self.diff2_x(T) + self.diff2_y(T) + self.diff2_z(T)))/rou/self.cv \
            - u * self.diff1_x(T) - v * self.diff1_y(T) - w * self.diff1_z(T)

        dc = -u * self.diff1_x(c) - v * self.diff1_y(c) - w * self.diff1_z(c)

        result = torch.stack(
            [du.squeeze(), dv.squeeze(), dw.squeeze(), drou.squeeze(), dT.squeeze(), dc.squeeze()], dim=0)

        # print("result shape:", result.shape)
        result[0:3, :, :, 0] = 0.0
        result[0:3, :, :, -1] = 0.0

        result[4, :, :, 0] = 0.0
        result[4, :, :, -1] = 0.0

        return result


def construct_initial_state():
    """
    construct the initial problem definition
    """
    nx, ny, nz = 50, 2, 50
    dx = 0.06
    x = torch.arange(0, nx * dx, dx, dtype=torch.float32)

    dy = 0.05
    y = torch.arange(0, ny * dy, dy, dtype=torch.float32)

    dz = 0.02
    z = torch.arange(0, nz * dz, dz, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y)

    grid_info = (x, y, z)

    u0_true = torch.ones(nx, ny, nz)*0.1
    v0_true = torch.zeros(nx, ny, nz)
    w0_true = torch.zeros(nx, ny, nz)

    # w0_true[nx//2, :, nz//2] = 0.1

    rou0_true = torch.ones(nx, ny, nz)
    T0_true = 0.1*torch.ones(nx, ny, nz)
    # T0_true[:, :, nz-1] = 1.0*0.1
    # T0_true[:, :, 0] = 2.0 * 0.1

    for index in range(nz):
        T0_true[:, :, index] = (2.0 - 1.0 * index / nz) * 0.1

    c0_true = torch.zeros(nx, ny, nz)
    c0_true[20:30, :, 20:30] = 1.0

    state = torch.stack(
        [u0_true, v0_true, w0_true, rou0_true, T0_true, c0_true], dim=0)

    cp, cv, mu, k, g = 1.2, 1.0, 1e-7, 1e-5, 0.1
    simu_para = cp, cv, mu, k, g

    return grid_info, simu_para, state


grid_info, simu_para, state_True = construct_initial_state()

(x, y, z) = grid_info
print(x, z)
device = "cpu"
# print(result)
pde_model = Rayleigh_Bernard_Model(grid_info, simu_para).to(device)

t = torch.linspace(0., 5.0, 200).to(device)
# %%
start_time = time.time()
y_true = odeint(pde_model, state_True, t, method=args.method).to(device)

print(y_true.mean(dim=(0, 2, 3, 4)))

show = y_true[-1, 4, :, 0, :].squeeze().cpu().detach().numpy()

fig, ax = fig, ax = plt.subplots(figsize=(10, 4))
# plt.plot(x, dudx.squeeze().cpu().detach().numpy(), '-x', markersize=2)
X, Z = np.meshgrid(z, x)
XX, ZZ = np.meshgrid(x, z)

contour = plt.contourf(XX,
                       ZZ,
                       np.einsum('ij->ji', show))
plt.axis('equal')
plt.colorbar()
# plt.contourf(show)

print(X.shape, Z.shape, show.shape)
plt.savefig("benard3.png")
