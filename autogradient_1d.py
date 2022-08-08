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

from dynamic_model.differ_module import Order2_Diff1, Order2_Diff2
from dynamic_model.integral_module import Euler_Method, RK4_Method
from neural_model.surrogate_module import Surrogate_Model_FC, Surrogate_Model_FC_Res, Surrogate_Model_CNN
from tools.statistical_helper import RunningAverageMeter
from tools.model_helper import ModelUtils
from tools.plot_helper import generate_line_movement_gif
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


def load_checkpoint(checkpoint, model, optimizer, optimizer_surrogate):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    model.state = checkpoint["state"]
    # 重新建立起 optimizer 和 model 之间的关系
    optimizer = torch.optim.Adam([model.state])
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_surrogate.load_state_dict(checkpoint["optimizer_surrogate"])
    return model, optimizer, optimizer_surrogate


class PDE_Model(nn.Module):
    def __init__(self, dx, zero_diff, first_diff, second_diff):
        super(PDE_Model, self).__init__()
        self.term1 = Order2_Diff1(dx, half_padding=1)
        self.term2 = Order2_Diff2(dx, half_padding=1)
        self.zero_diff = zero_diff
        self.first_diff = first_diff
        self.second_diff = second_diff

    def forward(self, u, zero_diff=(0, 1), first_diff=(0, 1), second_diff=(0, 1)):
        result = torch.zeros(u.shape)
        # u_clone = u.clone()
        # for term, value in enumerate(self.zero_diff):
        #     result = result + value*torch.pow(u_clone, term)*u

        # for term, value in enumerate(self.first_diff):
        #     result = result + value*torch.pow(u_clone, term)*self.term1(u)

        # for term, value in enumerate(self.second_diff):
        #     result = result + value*torch.pow(u_clone, term)*self.term2(u)

        result = (- (u + 0.5) * self.term1.forward(u) +
                  0.0001 * self.term2.forward(u))
        return result


class Pipeline_model(nn.Module):
    def __init__(self, state, time_method, surrogate_model, total_iteration,
                 obs_map={0: 100, 1: 200, 2: 300},
                 surrogate_steps=[],
                 surrogate_span=10,
                 surrogate_res=True
                 ):
        super(Pipeline_model, self).__init__()
        self.time_method = time_method
        self.total_iteration = total_iteration
        self.obs_map = obs_map
        self.surrogate_steps = surrogate_steps
        self.surrogate_span = surrogate_span
        self.surrogate = surrogate_model
        self.surrogate_res = surrogate_res
        self.state = state
        self.state.requires_grad = True

    def forward(self, u):
        obs_ = torch.zeros(len(self.obs_map), list(u.shape)[0]).to(device)
        index = 0
        iteration = 0
        while(iteration < self.total_iteration):
            if(iteration in self.surrogate_steps):
                u = u + self.surrogate(u)
                iteration = iteration + self.surrogate_span
            else:
                u = self.time_method(u)
                iteration = iteration + 1

            if (iteration in self.obs_map.values()):
                obs_[index, :] = 1.0*u
                index = index + 1
        return u, obs_


statMeter = RunningAverageMeter()


def construct_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.01
    x = torch.arange(0,  100*dx, dx, dtype=torch.float32)
    u = 1.0 + torch.sin(2*np.pi*x).to(torch.float32)

    u0_true = (1.0
               + 0.2 * torch.sin(2*np.pi * x).to(device)
               + 0.1*torch.cos(6*np.pi*x + 1.0/3.0).to(device)
               + 0.1*torch.sin(10*np.pi*x + 5.0/9.0).to(device)
               + 0.01 * torch.randn(list(x.shape)[0]).to(device))

    x, u, u0_true = x.to(device), u.to(device), u0_true.to(device)
    u.requires_grad = True

    return x, u, u0_true, dx


x, u, u0_true, dx = construct_initial_state()

# pde model
pde_model = PDE_Model(
    dx, zero_diff=(), first_diff=(-0.5, -1.0), second_diff=(1e-4,)).to(device)

# time forward
time_forward_method = RK4_Method(pde_model, dt=0.003).to(device)

# twin framework
with torch.no_grad():
    # assimulation_obs_map = {0: 100, 1: 200, 2: 300}
    assimulation_obs_map = {
        k: v for k, v in enumerate(np.arange(100, 303, 10))}
    observation_sigma = 0.01
    true_model = Pipeline_model(u.clone(), time_forward_method, surrogate_model=None,
                                total_iteration=300,
                                obs_map=assimulation_obs_map).to(device)
    y0_True, obs_True = true_model(u0_true)
    obs_True_noise = obs_True + observation_sigma * \
        torch.randn(obs_True.shape).to(device)
    u0_true.requires_grad = False

    ModelUtils.print_model_layer(true_model)
    logger.info(true_model)

# %%

surrogate_model = Surrogate_Model_FC_Res(
    n_features=100, hidden_dims=(), p=0.0).to(device)

# surrogate_model = Surrogate_Model_CNN(
#     input_channel=1, hidden_channels=(5, 5, 3), p=0.0).to(device)


logger.info(surrogate_model)
surrogate_steps = np.arange(100, 300, 10)
surrogate_steps = []
surrogate_span = 1
assimulation_obs_map = {
    k: v for k, v in enumerate(np.arange(100, 303, 10))}

# Pipeline_model中，state改成 u.clone() 则无法backward
pipeline_model = Pipeline_model(u, time_forward_method, surrogate_model,
                                total_iteration=300,
                                obs_map=assimulation_obs_map,
                                surrogate_steps=surrogate_steps,
                                surrogate_span=surrogate_span).to(device)

# %%
##################################
pipeline_model.obs_map = {
    k: v for k, v in enumerate(np.arange(100, 303, 10))}
optimizer = torch.optim.Adam([pipeline_model.state], lr=0.05)
optimizer_surrogate = torch.optim.Adam(
    pipeline_model.surrogate.parameters(), lr=0.0003)
criterion = torch.nn.MSELoss()


if(load_model):
    pipeline_model, optimizer, optimizer_surrogate = load_checkpoint(torch.load(checkpoint_file_name), pipeline_model,
                                                                     optimizer, optimizer_surrogate)

# %%


def base_observation_plot(fig_index):
    import matplotlib.pyplot as plt
    plt.figure(101)
    plt.plot(x.detach().numpy(), u.detach().numpy(),
             label="analysis_start")
    plt.plot(x, y_pred.detach(
    ).numpy(), label="analysis_end")
    plt.plot(x, u0_true.detach(
    ).numpy(), '-k', label="true_start")
    plt.plot(x, y0_True.detach(
    ).numpy(), label="true_end")
    plt.legend()

    # color map from https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = plt.get_cmap('tab20c')
    N_color = 10
    plt.figure(fig_index + 1)
    plt.plot(x, obs_pred.detach().numpy()[
        10, :], '-', color=cmap(float(1)/N_color), label="0.3")
    plt.plot(x, obs_pred.detach().numpy()[
        15, :], '-', color=cmap(float(2)/N_color), label="0.6")
    plt.plot(x, obs_pred.detach().numpy()[
        20, :], '-', color=cmap(float(3)/N_color), label="0.9")
    plt.legend()

    plt.figure(fig_index + 2)
    plt.plot(x, obs_True.detach().numpy()[
        10, :], '-', color=cmap(float(1)/N_color), label="0.3")
    plt.plot(x, obs_True.detach().numpy()[
        15, :], '-', color=cmap(float(2)/N_color), label="0.6")
    plt.plot(x, obs_True.detach().numpy()[
        20, :], '-', color=cmap(float(3)/N_color), label="0.9")
    plt.legend()

    plt.figure(fig_index + 3)
    plt.plot(x, obs_True_noise.detach().numpy()[
        0, :], '-', color=cmap(float(1)/N_color), label="0.3")
    plt.plot(x, obs_True_noise.detach().numpy()[
        1, :], '-', color=cmap(float(2)/N_color), label="0.6")
    plt.plot(x, obs_True_noise.detach().numpy()[
        2, :], '-', color=cmap(float(3)/N_color), label="0.9")
    plt.legend()


total_iteration = 2000
ploting_interval = 200

pipeline_model.train()
for iteration in range(total_iteration):
    time_start = time.time()

    y_pred, obs_pred = pipeline_model(pipeline_model.state)
    loss = criterion(obs_pred, obs_True_noise)

    optimizer.zero_grad()
    loss.backward()

    # print(torch.sum(torch.abs(pipeline_model.state.grad)))
    # print(torch.mean(u))

    optimizer.step()

    if(len(pipeline_model.surrogate_steps)):
        optimizer_surrogate.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_surrogate.step()

    writer.add_scalar("loss", loss.item(), global_step=iteration)
    logger.info("iteration: {}, loss:{}".format(iteration, loss.item()))
    logger.info("elapse time:{}".format(time.time() - time_start))
    logger.info("#"*20)

    torch.cuda.empty_cache()
    if (((iteration+1) % save_interval == 0) and save_model):
        checkpoint = {"state_dict": pipeline_model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "optimizer_surrogate": optimizer_surrogate.state_dict(),
                      "state": pipeline_model.state}

        ModelUtils.save_checkpoint(
            checkpoint, filename=save_checkpoint_file_name)

    if ((iteration+1) % ploting_interval) == 0:
        pass
        with torch.no_grad():
            pipeline_model.eval()

            pipeline_model.train()

# %%
pipeline_model.eval()
with torch.no_grad():
    pipeline_model.obs_map = {
        k: v for k, v in enumerate(np.arange(0, 303, 10))}
    true_model.obs_map = {
        k: v for k, v in enumerate(np.arange(0, 303, 10))}
    y0_plot, obs_plot = pipeline_model(u)
    y0_plot_true, obs_plot_true = true_model(u)

    x_np = x.detach().numpy()
    obs_plot_np = obs_plot.detach().numpy()
    obs_plot_true_np = obs_plot_true.detach().numpy()

generate_line_movement_gif(x=x_np, y=obs_plot_np, file_name='./result/result_pipeline_0812_step5.gif',
                           fps=20, xlim=(0, 1), ylim=(0.0, 2.5))
generate_line_movement_gif(x=x_np, y=obs_plot_true_np, file_name='./result/result_true_0812_step5.gif',
                           fps=20, xlim=(0, 1), ylim=(0.0, 2.5))

# %%

plot_model_comparison = True

if(plot_model_comparison):
    import matplotlib.pyplot as plt
    pipeline_model.eval()
    true_result = u0_true.clone()
    surrogate_result = u0_true.clone()
    pipeline_result = u0_true.clone()

    view_step = 10
    with torch.no_grad():
        for i in range(view_step):
            true_result = true_result + pipeline_model.time_method(true_result)

        for i in range(int(view_step/pipeline_model.surrogate_span)):
            surrogate_result = pipeline_model.surrogate(surrogate_result)

        pipeline_model.obs_map = {
            k: v for k, v in enumerate(np.arange(view_step, view_step+1, view_step))}
        _, obs_result = pipeline_model(pipeline_result)
    pipeline_model.train()
    plt.plot(x, u0_true.detach().numpy(), c="orange", label="previsou")
    plt.plot(x, true_result.detach().numpy(), c="red", label="true")
    plt.plot(x, surrogate_result.detach().numpy(), c='blue', label="surrogate")
    plt.plot(x, obs_result.detach().numpy()[0, :], c='green', label="mixture")
    plt.legend()


# %%
