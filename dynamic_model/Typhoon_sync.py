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
import scipy.stats as st
from .differ_module import Order2_Diff2_Unstructure, Order2_Diff1_Unstructure, Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from .initialize_tools import gkern
from neurodiffeq.solvers import Solver1D, Solver2D
from torch.optim import Adam


class Get_Laplace(nn.Module):
    def __init__(self, x_vector, y_vector):
        super().__init__()
        self.diff2_x = Order2_Diff2_Unstructure_Period(
            x_vector, total_dim=2, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure_Period(
            y_vector, total_dim=2, diff_dim=2)

        self.grad_x = Order2_Diff1_Unstructure_Period(
            x_vector, total_dim=2, diff_dim=1)
        self.grad_y = Order2_Diff1_Unstructure_Period(
            y_vector, total_dim=2, diff_dim=2)

    def forward(self, state):
        return self.diff2_x(state) + self.diff2_y(state)

    def getuv_from_phi(self, phi):
        return self.grad_y(phi), -self.grad_x(phi)

    def get_laplace_p(self, u, v):
        return -((self.grad_x(u)**2 + 2.0*self.grad_y(u)*self.grad_x(v) + self.grad_y(v)**2))**2


class Typhoon_sync(nn.Module):
    def __init__(self, grid_info):
        super().__init__()
        dx, dy, grid_x, grid_y, vector_x, vector_y = grid_info
        self.vector_x = vector_x
        self.vector_y = vector_y

        self.diff2_x = Order2_Diff2_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure_Period(
            self.vector_y, total_dim=2, diff_dim=2)

        self.grad_x = Order2_Diff1_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.grad_y = Order2_Diff1_Unstructure_Period(
            self.vector_y, total_dim=2, diff_dim=2)

        self.mu = 1e-4

    def forward(self, t, state):

        omega, phi, u, v, p, c = state
        omega, phi, u, v, p, c = omega.unsqueeze(
            0), phi.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0), c.unsqueeze(0)

        du = -u*self.grad_x(u) - v*self.grad_y(u) - \
            self.grad_x(p) + self.mu*self.diff2_x(u)
        dv = -u*self.grad_x(v) - v*self.grad_y(v) - \
            self.grad_y(p) + self.mu*self.diff2_x(v)

        dc = -u*self.grad_x(c) - v*self.grad_y(c)

        zeros_ = torch.zeros(du.shape)
        return torch.cat([zeros_, zeros_, du, dv, zeros_, dc], dim=0)

    def observe(self, state):
        return state

    def get_divergence(self, u, v):
        return self.grad_x(u) + self.grad_y(v)

    def get_laplace(self, p):
        result = self.diff2_x(p) + self.diff2_y(p)
        return result

    def get_pressuere(self, u, v):
        with torch.no_grad():
            div = self.get_divergence(u, v)

        pressure = torch.zeros(u.shape)
        pressure.requires_grad = True

        optimizer_p = Adam([pressure], lr=1e-4)
        criterion = nn.MSELoss()
        for i in range(100):  # 后续改成100
            la_pressure = self.get_laplace(pressure)
            loss = criterion(la_pressure, div)
            optimizer_p.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_p.step()
            print(f"inner iteration:{i}, loss:{loss}")

        return pressure

    def get_grad_x(self, p):
        return self.grad_x(p)

    def get_grad_y(self, p):
        return self.grad_y(p)

    def calculate_omega_from_uv(self, u, v):
        return self.grad_y(u) - self.grad_x(v)


def calcualte_omega_one(grid_x, grid_y, x0=1.0, y0=0.5, R0=0.2):
    omega = torch.zeros(grid_x.shape)
    # r = torch.sqrt((grid_x - (-0.5)) ** 2 + (grid_y - (-0.5)) ** 2)
    # omega += 50 * torch.exp(-(r / R0) ** 2) * (1 - (r ** 2))
    return omega


def calcualte_omega(grid_x, grid_y, x0=1.0, y0=0.5, R0=0.2):
    omega = torch.zeros(grid_x.shape)
    r = torch.sqrt((grid_x - (-0.5)) ** 2 + (grid_y - (-0.5)) ** 2)
    omega += 50*torch.exp(-(r/R0)**2)*(1-(r**2))

    r = torch.sqrt((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2)
    omega += 25 * torch.exp(-(r/R0)**2)*(1-(r**2))

    return omega


def calculate_phi_u_v_p(omega, vector_x, vector_y, grid_x, grid_y):
    phi = torch.zeros(omega.shape)
    phi.requires_grad = True

    model = Get_Laplace(vector_x, vector_y)
    optimizer = Adam([phi], lr=1e-3)

    criterion = nn.MSELoss()

    for i in range(300):
        la_phi = model(phi)
        optimizer.zero_grad()
        loss = criterion(la_phi, omega)
        loss.backward()
        optimizer.step()
        # print(loss)
        print(torch.mean(torch.abs(phi)))

    with torch.no_grad():
        u, v = model.getuv_from_phi(phi)
        u = u + 1.0

    with torch.no_grad():
        laplace_p = -model.get_laplace_p(u, v)

    pressure = torch.zeros(omega.shape)
    pressure.requires_grad = True
    optimizer_p = Adam([pressure], lr=1e-4)

    for i in range(1000):
        la_pressure = model(pressure)
        optimizer_p.zero_grad()
        loss = criterion(la_pressure, laplace_p)
        loss.backward()
        optimizer_p.step()
        print(loss)
    return phi.detach(), u, v, pressure.detach()


def get_concetraction(grid_x, grid_y):
    tt1 = torch.abs((grid_x/1.0) - torch.ceil(grid_x/1.0)) < 0.21
    tt2 = torch.abs((grid_y/1.0) - torch.ceil(grid_y/1.0)) < 0.21
    return 1.0*tt1*tt2


def construct_typhoonsync_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.1
    dy = 0.1
    Nx = 60
    Ny = 60
    vector_x = torch.arange(0,  Nx*dx, dx, dtype=torch.float32) - Nx*dx//2
    vector_y = torch.arange(0,  Ny*dy, dy, dtype=torch.float32) - Ny*dy//2

    grid_x, grid_y = torch.meshgrid(vector_x, vector_y)

    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y)

    omega = calcualte_omega(grid_x, grid_y, x0=0.0, y0=0.0, R0=1.0)
    omega = omega.unsqueeze(0)
    omega = omega.double()
    phi, u, v, p = calculate_phi_u_v_p(
        omega, vector_x, vector_y, grid_x, grid_y)
    concentration = get_concetraction(grid_x, grid_y).unsqueeze(0)
    state0_true = torch.cat([omega, phi, u, v, p, concentration], axis=0)

    omega_init = calcualte_omega_one(grid_x, grid_y, x0=0.0, y0=0.0, R0=1.0)
    omega_init = omega_init.unsqueeze(0)
    omega_init = omega_init.double()
    phi_init, u_init, v_init, p_init = calculate_phi_u_v_p(
        omega_init, vector_x, vector_y, grid_x, grid_y)
    concentration_init = get_concetraction(grid_x, grid_y).unsqueeze(0)
    state0_init = torch.cat(
        [omega_init, phi_init, u_init, v_init, p_init, concentration_init], axis=0)

    #### time info  ####
    nt_time = 300
    time_vector = torch.linspace(0., 0.01 * nt_time, nt_time + 1)

    ind_obs = torch.arange(30, 300, 30)
    nt_obs = len(ind_obs)

    time_info = (time_vector, nt_time, ind_obs, nt_obs)

    return grid_info, (state0_init, state0_true), time_info
