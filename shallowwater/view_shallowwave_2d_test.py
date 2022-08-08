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
from dynamic_model.differ_module import Order2_Diff1, Order2_Diff2
from dynamic_model.differ_module import Order2_Diff1_2D_X, Order2_Diff1_2D_Y, Order2_Diff2_2D_X, Order2_Diff2_2D_Y
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from tools.plot_helper import generate_image_movement_gif, generate_image_movement3d_gif

import numpy as np
import scipy.stats as st


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


class Args:
    method = "rk4"
    batch_size = 20
    niters = 1000
    viz = True
    adjoint = False
    load_model = False
    prefix = "ode2d_noadjoint_"
    checkpoint_file_name = "./checkpoint/ode/" + prefix + "test1.pth.tar"


args = Args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True

# %%


class Shallow_Water_Model(nn.Module):
    def __init__(self, grid_info):
        super(Shallow_Water_Model, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y = grid_info

        if(True):
            self.diff1_x = Order2_Diff1_2D_X(self.dx, half_padding=1)
            self.diff1_y = Order2_Diff1_2D_Y(self.dy, half_padding=1)
            self.diff2_x = Order2_Diff2_2D_X(self.dx, half_padding=1)
            self.diff2_y = Order2_Diff2_2D_Y(self.dy, half_padding=1)
        else:
            self.diff1_x = Order2_Diff1_Unstructure_Period(
                self.grid_x, total_dim=2, diff_dim=1)
            self.diff1_y = Order2_Diff1_Unstructure_Period(
                self.grid_x, total_dim=2, diff_dim=2)

            self.diff2_x = Order2_Diff2_Unstructure_Period(
                self.grid_y, total_dim=2, diff_dim=1)
            self.diff2_y = Order2_Diff2_Unstructure_Period(
                self.grid_y, total_dim=2, diff_dim=2)

        self.H = 1.0
        self.g = 9.8
        self.b = 0.1
        self.mu = 0.0001

    def forward(self, t, state):
        """
        shape [3,100,100]
        """
        h, u, v = state
        # print("h shape:.{}".format(h.shape))
        h, u, v = h.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0)

        dh = -1.0*self.diff1_x((self.H + h)*u) - 1.0 * \
            self.diff1_y((self.H + h)*v)

        du = self.mu*(self.diff2_x(u) + self.diff2_y(u)) \
            - self.g*self.diff1_x(h)
        -u*self.diff1_x(u) - v*self.diff1_y(u) - self.b*u

        dv = self.mu*(self.diff2_x(v) + self.diff2_y(v)) \
            - self.g*self.diff1_y(h)
        -u*self.diff1_x(v) - v*self.diff1_y(v) - self.b*v

        dh, du, dv = dh.squeeze(), du.squeeze(), dv.squeeze()
        result = torch.stack([dh, du, dv])
        return result


def construct_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.04
    dy = 0.04

    x = torch.arange(0,  25*dx, dx, dtype=torch.float32)
    y = torch.arange(0,  25*dy, dy, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y)

    grid_info = (dx, dy, grid_x, grid_y)
    # u0_true = torch.sin(2*np.pi * grid_x) + torch.sin(2*np.pi * grid_y)
    h0_true = torch.zeros(grid_x.shape)

    def fun1():
        # pertube_value1 = torch.tensor(gkern(7, 3))
        # h0_true[50:57, 50:57] = pertube_value1
        pertube_value2 = 2.0 * torch.tensor(gkern(6, 3))
        h0_true[15:21, 15:21] = pertube_value2

    def fun2():
        character_list = [[25, 46], [25, 47], [26, 46], [26, 48], [27, 46], [27, 48], [28, 46], [28, 49], [29, 45], [29, 46], [29, 49], [29, 50], [30, 45], [30, 50], [31, 44], [31, 50], [31, 51], [32, 44], [32, 51], [33, 44], [33, 51], [34, 43], [34, 52], [35, 43], [35, 52], [36, 42], [36, 52], [37, 42], [37, 53], [38, 41], [38, 53], [39, 40], [39, 54], [40, 40], [40, 54], [41, 40], [41, 55], [42, 39], [42, 55], [43, 39], [43, 55], [44, 39], [44, 55], [45, 38], [45, 56], [46, 38], [46, 56], [47, 38], [47, 56], [47, 57], [48, 37], [48, 57], [49, 37], [49, 57], [50, 36], [50, 57], [51, 36], [51, 58], [
            52, 35], [52, 58], [53, 35], [53, 58], [54, 35], [54, 37], [54, 38], [54, 39], [54, 40], [54, 41], [54, 42], [54, 43], [54, 44], [54, 45], [54, 46], [54, 47], [54, 48], [54, 49], [54, 50], [54, 51], [54, 52], [54, 53], [54, 54], [54, 55], [54, 58], [55, 34], [55, 55], [55, 56], [55, 57], [55, 58], [56, 34], [56, 58], [57, 34], [57, 58], [58, 33], [58, 59], [59, 33], [59, 59], [60, 33], [60, 59], [61, 32], [61, 33], [61, 59], [62, 32], [62, 59], [63, 32], [63, 60], [64, 31], [64, 32], [64, 60], [65, 31], [65, 60], [66, 30], [66, 31], [66, 60], [67, 30], [67, 60], [67, 61], [68, 61]]
        pertube_value1 = torch.tensor(gkern(7, 3))
        for x, y in character_list:
            h0_true[x - 3: x + 4, y - 3: y + 4] += pertube_value1

    def fun3():
        character_list = [[12, 69], [13, 17], [13, 18], [13, 19], [13, 20], [13, 21], [13, 22], [13, 69], [13, 70], [14, 17], [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [14, 27], [14, 28], [14, 29], [14, 69], [14, 70], [15, 17], [15, 29], [15, 30], [15, 31], [15, 32], [15, 33], [15, 68], [15, 69], [15, 70], [16, 17], [16, 34], [16, 35], [16, 36], [16, 68], [16, 70], [17, 17], [17, 36], [17, 37], [17, 38], [17, 67], [17, 71], [18, 17], [18, 38], [18, 39], [18, 40], [18, 67], [18, 71], [19, 17], [19, 40], [19, 41], [19, 66], [19, 71], [19, 72], [20, 17], [20, 41], [20, 42], [20, 66], [20, 72], [21, 17], [21, 43], [21, 44], [21, 65], [21, 72], [21, 73], [22, 17], [22, 44], [22, 65], [22, 73], [23, 17], [23, 44], [23, 45], [23, 64], [23, 65], [23, 73], [24, 17], [24, 45], [24, 46], [24, 64], [24, 73], [24, 74], [25, 17], [25, 46], [25, 47], [25, 64], [25, 74], [26, 17], [26, 47], [26, 48], [26, 49], [26, 63], [26, 64], [26, 74], [27, 17], [27, 49], [27, 50], [27, 63], [27, 74], [27, 75], [28, 17], [28, 50], [28, 62], [28, 63], [28, 75], [29, 17], [29, 51], [29, 62], [29, 75], [30, 17], [30, 51], [30, 62], [30, 75], [30, 76], [31, 17], [31, 52], [31, 61], [31, 62], [31, 76], [32, 17], [32, 52], [32, 53], [32, 61], [32, 76], [32, 77], [33, 17], [33, 53], [33, 60], [33, 61], [33, 77], [34, 17], [34, 54], [34, 60], [34, 77], [35, 17], [35, 54], [35, 60], [35, 77], [36, 17], [36, 54], [36, 55], [36, 59], [36, 77], [36, 78], [37, 17], [37, 55], [37, 59], [37, 73], [37, 78], [38, 17], [38, 25], [38, 49], [38, 55], [38, 58], [38, 59], [38, 69], [38, 70], [38, 71], [38, 72], [38, 73], [38, 74], [38, 75], [38, 76], [38, 77], [38, 78], [38, 79], [38, 80], [38, 81], [39, 17], [39, 24], [39, 25], [39, 49], [39, 55], [39, 58], [39, 67], [39, 68], [39, 69], [39, 78], [39, 81], [39, 82], [39, 83], [40, 17], [40, 24], [40, 25], [40, 48], [40, 49], [40, 55], [40, 58], [40, 65], [40, 66], [40, 67], [40, 78], [40, 79], [40, 83], [40, 84], [41, 17], [41, 24], [41, 25], [41, 48], [41, 49], [41, 55], [41, 58], [41, 63], [41, 64], [41, 79], [41, 85], [42, 17], [42, 24], [42, 25], [42, 48], [42, 49], [42, 55], [42, 57], [42, 58], [42, 61], [42, 62], [42, 63], [42, 79], [42, 85], [42, 86], [43, 17], [43, 23], [43, 26], [43, 47], [43, 50], [43, 55], [43, 57], [43, 60], [43, 61], [43, 79], [43, 80], [43, 86], [43, 87], [44, 17], [44, 23], [44, 26], [44, 47], [44, 50], [44, 55], [44, 56], [44, 57], [44, 59], [44, 60], [44, 80], [44, 87], [44, 88], [45, 17], [45, 23], [45, 26], [45, 27], [45, 47], [45, 50], [45, 55], [45, 56], [45, 58], [45, 59], [45, 80], [45, 88], [45, 89], [46, 17], [46, 23], [46, 27], [46, 46], [46, 51], [46, 55], [46, 56], [46, 57], [46, 58], [46, 80], [46, 89], [47, 17], [47, 23], [47, 27], [47, 46], [47, 51], [47, 55], [47, 57], [47, 81], [47, 89], [48, 17], [48, 22], [48, 28], [48, 45], [48, 46], [48, 51], [48, 55], [48, 56], [48, 81], [48, 90], [49, 17], [49, 22], [49, 28], [49, 45], [49, 52], [49, 55], [49, 56], [49, 81], [49, 90], [50, 17], [50, 22], [50, 28], [50, 29], [50, 44], [50, 45], [50, 52], [50, 53], [50, 54], [50, 55], [50, 57], [50, 58], [50, 59], [50, 60], [50, 61], [50, 62], [50, 63], [50, 64], [50, 65], [50, 82], [50, 90], [51, 17], [51, 22], [51, 29], [51, 44], [51, 53], [51, 54], [51, 55], [51, 66], [51, 67], [51, 68], [51, 69], [51, 70], [51, 71], [51, 72], [51, 73], [51, 74], [51, 75], [51, 76], [51, 77], [51, 78], [51, 79], [51, 80], [51, 82], [51, 90], [51, 91], [52, 17], [52, 22], [52, 29], [52, 44], [52, 53], [52, 54], [52, 55], [52, 82], [52, 91], [53, 17], [
            53, 21], [53, 30], [53, 43], [53, 54], [53, 82], [53, 83], [53, 91], [54, 17], [54, 21], [54, 30], [54, 43], [54, 54], [54, 83], [54, 91], [55, 17], [55, 21], [55, 30], [55, 43], [55, 53], [55, 54], [55, 83], [55, 91], [56, 17], [56, 20], [56, 30], [56, 31], [56, 42], [56, 43], [56, 52], [56, 53], [56, 54], [56, 83], [56, 84], [56, 91], [57, 17], [57, 20], [57, 31], [57, 42], [57, 52], [57, 54], [57, 55], [57, 84], [57, 91], [58, 17], [58, 20], [58, 31], [58, 32], [58, 42], [58, 51], [58, 52], [58, 54], [58, 55], [58, 84], [58, 92], [59, 17], [59, 19], [59, 32], [59, 42], [59, 50], [59, 51], [59, 52], [59, 54], [59, 55], [59, 84], [59, 92], [60, 17], [60, 19], [60, 32], [60, 41], [60, 42], [60, 49], [60, 50], [60, 51], [60, 52], [60, 53], [60, 54], [60, 56], [60, 85], [60, 92], [61, 17], [61, 18], [61, 33], [61, 41], [61, 48], [61, 49], [61, 51], [61, 53], [61, 56], [61, 85], [61, 92], [62, 17], [62, 18], [62, 33], [62, 41], [62, 47], [62, 48], [62, 50], [62, 51], [62, 53], [62, 56], [62, 86], [62, 92], [63, 17], [63, 18], [63, 33], [63, 40], [63, 41], [63, 46], [63, 47], [63, 50], [63, 53], [63, 57], [63, 86], [63, 92], [64, 17], [64, 18], [64, 33], [64, 40], [64, 45], [64, 50], [64, 53], [64, 57], [64, 86], [64, 92], [65, 17], [65, 34], [65, 40], [65, 43], [65, 44], [65, 49], [65, 53], [65, 58], [65, 87], [65, 91], [65, 92], [66, 17], [66, 34], [66, 39], [66, 40], [66, 41], [66, 42], [66, 43], [66, 49], [66, 53], [66, 58], [66, 87], [66, 91], [67, 17], [67, 34], [67, 38], [67, 39], [67, 40], [67, 41], [67, 48], [67, 53], [67, 58], [67, 87], [67, 91], [68, 16], [68, 17], [68, 34], [68, 35], [68, 36], [68, 37], [68, 38], [68, 39], [68, 47], [68, 48], [68, 53], [68, 59], [68, 87], [68, 88], [68, 91], [69, 16], [69, 17], [69, 33], [69, 34], [69, 35], [69, 39], [69, 47], [69, 53], [69, 59], [69, 88], [69, 91], [70, 16], [70, 17], [70, 30], [70, 31], [70, 32], [70, 33], [70, 34], [70, 39], [70, 47], [70, 53], [70, 59], [70, 88], [70, 89], [70, 91], [71, 15], [71, 16], [71, 26], [71, 27], [71, 28], [71, 29], [71, 30], [71, 35], [71, 39], [71, 46], [71, 53], [71, 60], [71, 90], [71, 91], [72, 15], [72, 16], [72, 23], [72, 24], [72, 25], [72, 26], [72, 35], [72, 38], [72, 46], [72, 53], [72, 54], [72, 60], [72, 90], [73, 15], [73, 16], [73, 17], [73, 18], [73, 19], [73, 20], [73, 21], [73, 22], [73, 35], [73, 38], [73, 54], [73, 60], [73, 61], [73, 90], [74, 14], [74, 35], [74, 38], [74, 54], [74, 61], [74, 89], [75, 14], [75, 35], [75, 38], [75, 54], [75, 61], [75, 89], [76, 14], [76, 35], [76, 36], [76, 37], [76, 54], [76, 61], [76, 88], [77, 13], [77, 14], [77, 36], [77, 37], [77, 54], [77, 55], [77, 61], [77, 62], [77, 88], [78, 13], [78, 36], [78, 37], [78, 54], [78, 55], [78, 62], [78, 88], [79, 13], [79, 36], [79, 54], [79, 62], [79, 87], [79, 88], [80, 13], [80, 36], [80, 54], [80, 55], [80, 62], [80, 63], [80, 87], [81, 12], [81, 13], [81, 36], [81, 55], [81, 63], [81, 86], [81, 87], [82, 12], [82, 55], [82, 56], [82, 63], [82, 86], [83, 12], [83, 56], [83, 57], [83, 63], [83, 64], [83, 85], [83, 86], [84, 11], [84, 12], [84, 57], [84, 58], [84, 59], [84, 64], [84, 83], [84, 84], [85, 11], [85, 59], [85, 60], [85, 61], [85, 64], [85, 82], [85, 83], [86, 11], [86, 61], [86, 62], [86, 64], [86, 81], [86, 82], [87, 10], [87, 11], [87, 63], [87, 64], [87, 65], [87, 66], [87, 79], [87, 80], [87, 81], [88, 10], [88, 65], [88, 66], [88, 67], [88, 68], [88, 69], [88, 70], [88, 71], [88, 72], [88, 73], [88, 74], [88, 75], [88, 76], [88, 77], [88, 78], [88, 79], [89, 66]]
        pertube_value1 = torch.tensor(gkern(7, 3))
        for x, y in character_list:
            h0_true[x - 3: x + 4, y - 3: y + 4] += pertube_value1

    fun1()

    u0_true = torch.zeros(grid_x.shape)
    v0_true = torch.zeros(grid_x.shape)

    state0_true = torch.stack([h0_true, u0_true, v0_true]).to(device)
    state = torch.zeros(state0_true.shape).to(device)
    state.requires_grad = True
    return grid_info, state, state0_true


(dx, dy, grid_x, grid_y), state, state0_true = construct_initial_state()
print(state.shape, state0_true.shape)

pde_model = Shallow_Water_Model(grid_info=(dx, dy, grid_x, grid_y)).to(device)

t = torch.linspace(0., 0.2, 20).to(device)
# %%
start_time = time.time()
y_true = odeint(pde_model, state0_true, t, method=args.method).to(device)
print("elapse time:{:.7f}".format(time.time() - start_time))

generate_image_movement_gif(
    y_true[:, 0, :, :].squeeze().detach().numpy(), "./results/shallow_wave_0921_right.gif", fps=10)

exit()


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    state = checkpoint["state"]
    return state


if(args.load_model):
    state = load_checkpoint(torch.load(
        args.checkpoint_file_name, map_location=torch.device('cpu')))
    state.requires_grad = True


y_pred = odeint(pde_model, state, t, method=args.method).to(device)

y_pred_repalce = torch.zeros(y_true.shape)

# state_index = np.arange(0, 1, 1)
# obs_index, state_index, 25: 35: 1, 0: 100: 1]


generate_image_movement_gif(
    y_true[:, 0, :, :].squeeze().detach().numpy(), "./results/shallow_wave_0830_right.gif", fps=10)

# generate_image_movement_gif(
#     y_true[:, 1, :, :].squeeze().detach().numpy(), "./result/shallow_wave_show_A_u.gif", fps=10)

# generate_image_movement_gif(
#     y_true[:, 2, :, :].squeeze().detach().numpy(), "./result/shallow_wave_show_A_v.gif", fps=10)
# %%
# plt.cla()
# plt.imshow(y_true[0, 0, :, :].detach().numpy())
# plt.savefig("./result/view_assi_A.png")

# %%
# Plot the surface.
# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py


# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# surf = ax.plot_surface(grid_x, grid_y, y_pred[0, 0, :, :].squeeze().detach().numpy(), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.set_zlim(-0.01, 0.04)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.8, aspect=10)

# # plt.show()
# plt.savefig("./result/3d_view.png")
# generate_image_movement3d_gif(y_true[:, 0, :, :].squeeze().detach().numpy(),
#                               grid_x, grid_y, "./result/3d_view_A.gif", fps=10)
