#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen
"""
# %%
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
import logging

from collections import OrderedDict
import os
import time
from torch.utils.tensorboard import SummaryWriter

from neural_model.differ_module import Order2_Diff1, Order2_Diff2
from neural_model.integral_module import RK4_Method
from tools.statistical_helper import RunningAverageMeter
from tools.model_helper import ModelUtils
# from tools.plot_helper import generate_line_movement_gif
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# for debug purpose
torch.autograd.set_detect_anomaly(False)

# set logging
filehandler = logging.FileHandler("logs/autogradient/auto_assimulate.log")
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)


time_str = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
writer = SummaryWriter("runs/autogradient/" + time_str)

# random variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True
logger.info(device)

load_model = False
save_model = True
checkpoint_file_name = "./checkpoint/saved_checkpoint_debug.pth.tar"
save_checkpoint_file_name = "saved_checkpoint_debug.pth.tar"
save_interval = 5


class Lorenz63(nn.Module):  # Lorenz 96 model
    def __init__(self, *args):
        super(Lorenz63, self).__init__()
        self.sigma = args[0]
        self.beta = args[1]
        self.rho = args[2]

    def forward(self, state):
        # Unpack the state vector
        x, y, z = state[0], state[1], state[2]
        f = torch.zeros(state.shape)  # Derivatives
        f[0] = self.sigma * (y - x)
        f[1] = x * (self.rho - z) - y
        f[2] = x * y - self.beta * z
        return f


# pde model
sigma = 10.0
beta = 8.0/3.0
rho = 28.0


class Pipeline_model(nn.Module):
    def __init__(self, state, time_method, total_iteration,
                 obs_map={0: 100, 1: 200, 2: 300}
                 ):
        super(Pipeline_model, self).__init__()
        self.time_method = time_method
        self.total_iteration = total_iteration
        self.obs_map = obs_map
        self.state = state
        self.state.requires_grad = True

    def forward(self, u):
        obs_ = torch.zeros(len(self.obs_map), list(u.shape)[0]).to(device)
        index = 0
        iteration = 0
        while(iteration < self.total_iteration):
            u = self.time_method(u)
            iteration = iteration + 1

            if (iteration in self.obs_map.values()):
                obs_[index, :] = 1.0*u
                index = index + 1
        return u, obs_


statMeter = RunningAverageMeter()


u0_true = torch.tensor([1.0, 1.0, 1.0]).to(device)
print(u0_true.shape)
# u = torch.tensor([3.0, 4.0, 5.0]).to(device)

u = torch.tensor([1.0 + 1.0e-5,  1.0 + 1.0e-5,  1.0 + 1.0e-5]).to(device)
u.requires_grad = True

# pde model
pde_model = Lorenz63(sigma, beta, rho).to(device)

# time forward
time_forward_method = RK4_Method(pde_model, dt=0.01).to(device)

# twin framework
with torch.no_grad():
    # assimulation_obs_map = {0: 100, 1: 200, 2: 300}
    assimulation_obs_map = {
        k: v for k, v in enumerate(np.arange(100, 401, 20))}
    true_model = Pipeline_model(u.clone(), time_forward_method,
                                total_iteration=400,
                                obs_map=assimulation_obs_map).to(device)
    y0_True, obs_True = true_model(u0_true)

    # obs_True_noise = obs_True + observation_sigma * \
    #     torch.randn(obs_True.shape).to(device)
    u0_true.requires_grad = False

    ModelUtils.print_model_layer(true_model)
    logger.info(true_model)


# %%

assimulation_obs_map = {
    k: v for k, v in enumerate(np.arange(100, 401, 20))}

# Pipeline_model中，state改成 u.clone() 则无法backward
pipeline_model = Pipeline_model(u, time_forward_method,
                                total_iteration=400,
                                obs_map=assimulation_obs_map).to(device)

# %%
##################################

optimizer = torch.optim.Adam([pipeline_model.state], lr=0.1)
criterion = torch.nn.MSELoss()


total_iteration = 1  # 2000
ploting_interval = 200

pipeline_model.train()
for iteration in range(total_iteration):
    time_start = time.time()

    y_pred, obs_pred = pipeline_model(pipeline_model.state)
    loss = 3*criterion(obs_pred, obs_True)

    optimizer.zero_grad()
    loss.backward()
    print(f"grad:{1.0*pipeline_model.state.grad}")

    optimizer.step()

    writer.add_scalar("loss", loss.item(), global_step=iteration)
    logger.info("iteration: {}, loss:{}".format(iteration, loss.item()))
    logger.info("elapse time:{}".format(time.time() - time_start))
    logger.info("#"*20)

    torch.cuda.empty_cache()

    assimilation_loss = torch.mean(torch.abs(u - u0_true))
    assimilation_loss2 = torch.mean(torch.abs(obs_pred - obs_True))

    print(u)
    print("assimilaiton loss: {},ass_obs loss:{}".format(
        assimilation_loss.cpu().detach().numpy(),
        assimilation_loss2.cpu().detach().numpy()))
