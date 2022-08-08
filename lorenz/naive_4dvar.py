#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""

# %%
import os
import sys
sys.path.append("..") 
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from dynamic_model import time_integral, load_model, Lorenz
from dynamic_model.time_integral import Euler, RK4, JEuler, JRK4
import time

# set logging
filehandler = logging.FileHandler("../logs/4dvar/log.txt")
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# random variable
np.random.seed(seed=2)


class TimeConfig(object):
    """
    define the time infomation of the simulation and the observation
    1. sim_index : simulation index
    2. obs_index : observation index
    3. delta_t : delta time of the simulation
    """

    def __init__(self, simu_length, obs_index, delta_t):
        self.simu_index = np.arange(simu_length + 1)
        self.obs_index = obs_index
        self.delta_t = delta_t
        self.simu_series = self.simu_index * delta_t
        self.obs_series = obs_index * delta_t
        self.simu_length = simu_length
        self.obs_length = len(obs_index)


class DynamicEnv(object):
    def __init__(self, model, model_args, initial_state, time_config: TimeConfig, time_scheme):
        self.model = model
        self.model_args = model_args
        self.initial_state = initial_state
        self.time_config = time_config
        self.time_scheme = time_scheme
        self.state_dim = initial_state.shape[0]
        self.state = np.zeros([self.state_dim, time_config.simu_length + 1])
        self.state[:, 0] = initial_state
        self.obs = None

    def simulate(self):
        """
        simulate_forward
        """
        for step in range(self.time_config.simu_length):
            self.state[:, step + 1] = self.time_scheme(self.model,
                                                       self.state[:, step],
                                                       self.time_config.delta_t,
                                                       *self.model_args)
        pass

    def generate_obs(self):
        """
        generate observation from state, to be modified
        """
        obs_dim = 3
        sig_m = 0.00
        self.obs = np.zeros([obs_dim, self.time_config.obs_length])
        self.obs = self.state[0:obs_dim, self.time_config.obs_index] +  \
            np.random.normal(0, sig_m, [obs_dim, self.time_config.obs_length])
        pass


# model definition
dynamic_model, u0True, dynamic_model_args, J_dynamic_model = load_model.load_Lorenz63()

# time config
time_config = TimeConfig(
    simu_length=400, obs_index=np.arange(100, 401, 20).astype(int), delta_t=0.01)

# time scheme definition
time_scheme = time_integral.RK4


def plot_Lorenz63(time, state, time_obs, obs, fig_index):
    """
    plot the basic figure of the Lorenz63 model
    """
    fig = plt.figure(fig_index, figsize=(6, 4))
    plt.plot(time, state[0, :], c="tab:blue")
    plt.plot(time, state[1, :], c="tab:orange")
    plt.plot(time, state[2, :], c="tab:green")
    plt.plot(time_obs, obs[0, :], 'x', c="tab:blue")
    plt.plot(time_obs, obs[1, :], 'x', c="tab:orange")
    plt.plot(time_obs, obs[2, :], 'x', c="tab:green")
    plt.xlabel("time")
    plt.ylabel("state value")

    fig = plt.figure(fig_index + 1, figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state[0, :], state[1, :], state[2, :])
    pass


dynamicEnv = DynamicEnv(
    dynamic_model, dynamic_model_args, u0True, time_config, time_scheme)

dynamicEnv.simulate()
dynamicEnv.generate_obs()

plot_Lorenz63(dynamicEnv.time_config.simu_series, dynamicEnv.state,
              dynamicEnv.time_config.obs_series, dynamicEnv.obs, fig_index=0)


# dynamicEnv = DynamicEnv(
#     dynamic_model, dynamic_model_args, u0True, time_config, time_scheme)

# dynamicEnv.simulate()
# dynamicEnv.generate_obs()
# plot_Lorenz63(dynamicEnv.time_config.simu_series, dynamicEnv.state,
#               dynamicEnv.time_config.obs_series, dynamicEnv.obs, fig_index=10)

# %%

# ##########  data assimilation  ##########
R = np.eye(3)*0.05**2
B = np.eye(3)*0.1**2
H = np.eye(3, 3)


def Lin3dvar(x_b, obs, R, H, B):
    """
    model for 3dvar
    n: variable space,
    m: observation space
    solve Ax = b problem
    """
    B_inv = np.linalg.inv(B)
    R_inv = np.linalg.inv(R)
    A = B_inv + (H.T) @ R_inv @ H
    b = B_inv @ x_b - (H.T) @ R_inv @ obs
    return A, b


def Adj4dvar(rhs, Jrhs, ObsOp, JObsOp, dynamic_env: DynamicEnv, x_b, obs, R, args):
    """
    solver 4dvar by adjoint model
    """
    nt = dynamic_env.time_config.obs_index[-1]
    dt = dynamic_env.time_config.delta_t
    ind_m = dynamic_env.time_config.obs_index
    state = np.zeros([dynamic_env.state_dim, nt+1])  # base trajectory
    lam = np.zeros([dynamic_env.state_dim, nt+1])  # lambda sequence
    fk = np.zeros([dynamic_env.state_dim, dynamic_env.time_config.obs_length])

    Ri = np.linalg.inv(R)
    state[:, 0] = x_b

    # forward model
    for k in range(nt):
        state[:, k+1] = RK4(rhs, state[:, k], dt, *args)

    # backward adjoint model

    k = ind_m[-1]
    fk[:, -1] = (JObsOp(state[:, k])
                 ).T @ Ri @ (obs[:, -1] - ObsOp(state[:, k]))
    lam[:, k] = fk[:, -1]  # lambda_N = f_N

    km = len(ind_m)-2
    for k in range(ind_m[-1], 0, -1):
        DM = JRK4(rhs, Jrhs, state[:, k-1], dt, *args)
        lam[:, k-1] = (DM).T @ lam[:, k]
        if k-1 == ind_m[km]:
            fk[:, km] = (JObsOp(state[:, k-1])
                         ).T @ Ri @ (obs[:, km] - ObsOp(state[:, k-1]))
            lam[:, k-1] = lam[:, k-1] + fk[:, km]
            km = km - 1

    dJ0 = -lam[:, 0]
    return dJ0


def linear_solve(A, b):
    return np.linalg.solve(A, b)


def torch_solver(A, b):
    """
    solver Ax = b using pytorch,
    arguments to be added , lr,  iteration
    """
    dim = A.shape[0]
    x = torch.rand(dim, 1).double()
    x.requires_grad = True
    delta = torch.mm(A, x) - b
    loss = torch.norm(delta, p=2)
    optimizer = torch.optim.SGD([x], lr=3.0e-4)
    for i in range(200):
        delta = torch.mm(A, x) - b
        loss = torch.norm(delta, p=2)
        # logging.info(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return x.detach().numpy()


def h(u):
    w = u
    return w


def Dh(u):
    n = len(u)
    D = np.eye(n)
    return D


x_back = np.asarray([3.0, 4.0, 5.0])
xa = x_back

iteration_N = 100
intermediate_analysis = np.zeros([3, iteration_N+1])
intermediate_analysis[:, 0] = x_back
for i in range(iteration_N):
    dJ0 = Adj4dvar(dynamic_model, J_dynamic_model, h, Dh, dynamicEnv,
                   xa, dynamicEnv.obs, R,
                   dynamicEnv.model_args)
    print(dJ0)
    p = -dJ0 / np.linalg.norm(dJ0)
    xa = xa + 0.1 * p
    assimilation_loss = np.mean(np.abs(xa - u0True))
    intermediate_analysis[:, i + 1] = xa

    dynamicEnv_temp = DynamicEnv(
        dynamic_model, dynamic_model_args, xa, time_config, time_scheme)
    dynamicEnv_temp.simulate()
    dynamicEnv_temp.generate_obs()

    # print(dynamicEnv.obs)
    assimilation_loss2 = np.mean(
        np.abs(np.asarray(dynamicEnv_temp.obs) - np.asarray(dynamicEnv.obs)))

    print(xa, assimilation_loss, assimilation_loss2)


plt.plot(np.arange(iteration_N+1), intermediate_analysis[0, :])
plt.plot(np.arange(iteration_N+1), intermediate_analysis[1, :])
plt.plot(np.arange(iteration_N+1), intermediate_analysis[2, :])
plt.xlabel("iteration")
plt.ylabel("analysis values")


dynamicEnv2 = DynamicEnv(
    dynamic_model, dynamic_model_args, x_back, time_config, time_scheme)
dynamicEnv2.simulate()
plot_Lorenz63(dynamicEnv2.time_config.simu_series, dynamicEnv2.state,
              dynamicEnv.time_config.obs_series, dynamicEnv.obs, fig_index=2)


dynamicEnv3 = DynamicEnv(
    dynamic_model, dynamic_model_args, xa, time_config, time_scheme)
dynamicEnv3.simulate()
plot_Lorenz63(dynamicEnv3.time_config.simu_series, dynamicEnv3.state,
              dynamicEnv.time_config.obs_series, dynamicEnv.obs, fig_index=4)

# print(dJ0)
# %%
print("##"*50)
print("the below about 3dvar")

R = np.eye(3)*0.05**2
B = np.eye(3)*0.1**2
H = np.eye(3, 3)
A, b = Lin3dvar(np.expand_dims(dynamicEnv.state[:, 0], 1),
                np.expand_dims(dynamicEnv.obs[:, 0], 1), R, H, B)

print(A.shape, b.shape)


#  test of 3dvar
time_start = time.time()
ua = linear_solve(A, b)
logger.info("elapse time:{}".format(time.time() - time_start))
print(ua)

time_start = time.time()
ua = torch_solver(torch.from_numpy(A).double(), torch.from_numpy(b).double())
logger.info("elapse time:{}".format(time.time() - time_start))
result = ua
print(result)
