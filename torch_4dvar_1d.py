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
import logging
from neural_model.differ_module import Order2_Diff1, Order2_Diff2
from torch.utils.tensorboard import SummaryWriter

from tools.statistical_helper import RunningAverageMeter
from tools.model_helper import ModelUtils
from tools.plot_helper import generate_line_movement_gif
from tools.file_helper import FileUtils

# for debug purpose
torch.autograd.set_detect_anomaly(False)


class Args:
    method = "rk4"
    batch_size = 20
    niters = 1000
    viz = True
    adjoint = True
    load_model = False
    save_model = True
    prefix = "ode1d_adjoint_"
    checkpoint_file_name = "./checkpoint/ode/" + prefix + "test1.pth.tar"
    save_checkpoint_file_name = "./checkpoint/ode/" + prefix + "test1.pth.tar"
    save_interval = 50
    plot_interval = 50


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

random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)

time_str = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
writer = SummaryWriter("runs/ode/" + args.prefix + time_str)

# random variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True


class PDE_Model(nn.Module):
    def __init__(self, dx, zero_diff, first_diff, second_diff):
        super(PDE_Model, self).__init__()
        self.term1 = Order2_Diff1(dx, half_padding=1)
        self.term2 = Order2_Diff2(dx, half_padding=1)
        self.zero_diff = zero_diff
        self.first_diff = first_diff
        self.second_diff = second_diff

    def forward(self, t, u, zero_diff=(0, 1), first_diff=(0, 1), second_diff=(0, 1)):

        if(False):
            result = torch.zeros(u.shape)
            u_clone = u.clone()
            for term, value in enumerate(self.zero_diff):
                result = result + value*torch.pow(u_clone, term)*u

            for term, value in enumerate(self.first_diff):
                result = result + value*torch.pow(u_clone, term)*self.term1(u)

            for term, value in enumerate(self.second_diff):
                result = result + value*torch.pow(u_clone, term)*self.term2(u)

        result = torch.zeros(u.shape)
        result = (- (u + 0.5) * self.term1.forward(u) +
                  0.0001 * self.term2.forward(u))
        return result


def construct_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.01
    x = torch.arange(0, 100 * dx, dx, dtype=torch.float32)

    grid_info = (dx, x)

    u = 1.0 + torch.sin(2 * np.pi * x).to(torch.float32)

    u0_true = 1.0 \
        + 0.2 * torch.sin(2*np.pi * x) \
        + 0.1*torch.cos(6*np.pi*x + 1.0/3.0) \
        + 0.1*torch.sin(10*np.pi*x + 5.0/9.0) \
        + 0.01 * torch.randn(list(x.shape)[0])

    u = u.unsqueeze(0).unsqueeze(0)
    u.requires_grad = True
    u0_true = u0_true.unsqueeze(0).unsqueeze(0)
    return grid_info, u, u0_true


(dx, x), u, u0_true = construct_initial_state()
x, u, u0_true = x.to(device), u.to(device), u0_true.to(device)


def plot_result(x, u0_true, y_true, u, y_pred, file_name):  # %%
    if (args.viz):
        import matplotlib.pyplot as plt
        print("=> ploting_result")
        with torch.no_grad():
            plt.figure(0)
            plt.clf()
            plt.title("")
            plt.plot(x.cpu().detach().numpy(), u0_true.cpu().squeeze(
            ).detach().numpy(), '-k', label="true_start")
            plt.plot(x.cpu().detach().numpy(), y_true[-1, 0, :].cpu().squeeze().detach(
            ).numpy(), label="true_end")

            plt.plot(x.cpu().detach().numpy(), u.cpu().squeeze(
            ).detach().numpy(), 'tab:gray', label="pred_start")
            plt.plot(x.cpu().detach().numpy(), y_pred[-1, 0, :].cpu().squeeze().detach(
            ).numpy(), label="pred_end")
            plt.legend()
            plt.tight_layout()
            folder_name = "./result/wave_1d/"
            FileUtils.makedir(folder_name)
            plt.savefig(os.path.join(folder_name, file_name))


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    u = checkpoint["state"]
    return u


if(args.load_model):
    u = load_checkpoint(torch.load(args.checkpoint_file_name))
    u.requires_grad = True

# pde model
pde_model = PDE_Model(
    dx, zero_diff=(), first_diff=(-0.5, -1.0), second_diff=(1e-4,)).to(device)
t = torch.linspace(0., 0.9, 300).to(device)

# %%
y_true = odeint(pde_model, u0_true, t, method=args.method).to(device)
# %%
optimizer = torch.optim.Adam([u], lr=0.03)
criterion = torch.nn.MSELoss()
obs_index = np.arange(200, 300, 10)

for iteration in range(args.niters):

    time_start = time.time()
    y_pred = odeint(pde_model, u, t, method=args.method).to(device)
    optimizer.zero_grad()
    loss = criterion(y_pred[obs_index], y_true[obs_index])
    loss.backward(retain_graph=True)
    optimizer.step()

    writer.add_scalar("loss", loss.item(), global_step=iteration)
    logger.info("iteration: {}, loss:{}".format(iteration, loss.item()))
    logger.info("elapse time:{}".format(time.time() - time_start))

    if (((iteration+1) % args.save_interval == 0) and args.save_model):
        checkpoint = {"state": u}
        ModelUtils.save_checkpoint(
            checkpoint, filename=args.save_checkpoint_file_name)

    if (((iteration+1) % args.plot_interval == 0)):
        file_name = args.prefix + "result" + str(iteration+1).zfill(4) + ".png"
        plot_result(x, u0_true, y_true, u, y_pred, file_name)
        # %%
