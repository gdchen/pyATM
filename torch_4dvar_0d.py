#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""
# %%
import torch
import torch.nn as nn
import time
import logging
import numpy as np


class Args:
    method = "rk4"
    data_size = 1000
    batch_time = 10
    batch_size = 20
    niters = 2000
    test_freq = 20
    viz = True
    gpu = 0
    adjoint = False
    prefix = "ode0d_lorenz63"


args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


filehandler = logging.FileHandler(
    "logs/ode/" + args.prefix + "_assimulate.log")
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


# random variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True


def plot_Lorenz63(time, state, time_obs, obs, fig_index):
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


class Lorenz63(nn.Module):  # Lorenz 96 model
    def __init__(self, *args):
        super(Lorenz63, self).__init__()
        self.sigma = args[0]
        self.beta = args[1]
        self.rho = args[2]

    def forward(self, t, state):
        # Unpack the state vector
        x, y, z = state[0, 0], state[0, 1], state[0, 2]
        f = torch.zeros(state.shape)  # Derivatives
        f[0, 0] = self.sigma * (y - x)
        f[0, 1] = x * (self.rho - z) - y
        f[0, 2] = x * y - self.beta * z
        return f


# pde model
sigma = 10.0
beta = 8.0/3.0
rho = 28.0
pde_model = Lorenz63(sigma, beta, rho).to(device)
# time forward

with torch.no_grad():
    true_u0 = torch.tensor([[1.0, 1.0, 1.0]]).to(device)

u = torch.tensor([[3.0, 4.0, 5.0]]).to(device)
u.requires_grad = True

t = torch.linspace(0., 4.0, 401).to(device)
obs_index = np.arange(100, 401, 20)

true_y = odeint(pde_model, true_u0, t, method='rk4').to(device)

print(true_u0.shape, true_y.shape)

optimizer = torch.optim.Adam([u], lr=0.3)
criterion = torch.nn.MSELoss()

for iteration in range(200):
    time_start = time.time()
    pred_y = odeint(pde_model, u, t, method='rk4').to(device)
    optimizer.zero_grad()
    loss = criterion(pred_y[obs_index], true_y[obs_index])
    loss.backward(retain_graph=True)
    optimizer.step()

    print(u)
    assimilation_loss = torch.mean(torch.abs(u - true_u0))

    assimilation_loss2 = torch.mean(
        torch.abs(pred_y[obs_index] - true_y[obs_index]))

    logger.info("iteration: {:04d} | loss:{:.6f} |  ass_loss:{:.6f} | assobs_loss:{:.6f} | elapse time:{:.3f}".format(
        iteration, loss.item(), assimilation_loss.cpu().detach().numpy(),
        assimilation_loss2.cpu().detach().numpy(), time.time() - time_start))


# %%
# import matplotlib.pyplot as plt
# true_y_np = true_y.detach().numpy()
# pred_y_np = pred_y.detach().numpy()

# fig = plt.figure(1, figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(true_y_np[:, 0, 0], true_y_np[:, 0, 1], true_y_np[:, 0, 2])
# ax.plot(pred_y_np[:, 0, 0], pred_y_np[:, 0, 1], pred_y_np[:, 0, 2])


# result = time_forward_method(u0True)
# print(result)
# state_save = torch.zeros([3, 400])


# for i in range(400):
#     u0True = time_forward_method.forward(u0True)
#     state_save[:, i] = u0True

# fig = plt.figure(100 + 1, figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(state_save[0, :], state_save[1, :], state_save[2, :])


# %%
