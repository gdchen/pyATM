# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append("..") 
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
import time
from torch.optim import Adam
from dynamic_model.initialize_tools import gkern
from dynamic_model.Transport import Basic_Transport_Model, transport_simulation

"""
"""
class Args:
    method = "rk4"
    # niters = 1000
    # viz = True
    adjoint = False
    # load_model = False
    # prefix = "ode2d_noadjoint_"
    # checkpoint_file_name = "./checkpoint/ode/" + prefix + "test1.pth.tar"

args = Args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def plot_image(data, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(data[:, :], aspect='auto',
                   cmap='jet', animated=True)
    fig.colorbar(im)
    plt.savefig(filename, bbox_inches='tight')


def construct_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.02
    dy = 0.01
    x = torch.arange(0,  150 * dx, dx, dtype=torch.float32)
    y = torch.arange(0,  100 * dy, dy, dtype=torch.float32)

    grid_x, grid_y = torch.meshgrid(x, y ,indexing='ij')
    vector_x, vector_y = x, y

    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y)

    flux0 = torch.zeros(grid_x.shape)
    flux0[13:20,47:54] = 2.0 * torch.tensor(gkern(7, 4))
    concent0 = torch.zeros(grid_x.shape)*y
    u0 = torch.ones(grid_x.shape) + 0.5* torch.cos(grid_x)
    v0 = 0.5*grid_y
    state0 = torch.stack([flux0, concent0, u0, v0])
    return grid_info, state0


def construct_cuv_all(state0, time_vector):
    u0 = torch.clone(state0)[2,:,:]
    u_all_list = list()

    v0 = torch.clone(state0)[3,:,:]
    v_all_list = list()

    f0 = torch.clone(state0)[0,:,:]
    f_all_list = list()
    for i,value in enumerate(time_vector):
        u_all_list.append(u0* torch.sin(value) + 0.1*torch.randn(u0.shape))
        v_all_list.append(v0* torch.sin(value) + 0.1*torch.randn(v0.shape))
        f_all_list.append(f0 * torch.sin(value))

    u_all = torch.stack(u_all_list)
    v_all = torch.stack(v_all_list)
    c_all = state0[1,:,:].repeat([len(time_vector), 1,1])
    f_all = torch.stack(f_all_list)

    return f_all, c_all, u_all, v_all


#%% 坐标, flux, c, u, v 时间序列
(dx, dy, grid_x, grid_y, vector_x,
 vector_y), state0= construct_initial_state()
print(state0.shape, grid_x.shape)
time_vector = torch.linspace(0., 1.0, 101)
f_all, c_all, u_all, v_all = construct_cuv_all(state0, time_vector)
state_all = torch.stack([f_all, c_all, u_all, v_all]).permute(1,0,2,3)


# 读入模式开始更新
model = Basic_Transport_Model(grid_info=(
    dx, dy, grid_x, grid_y, vector_x, vector_y))
with torch.no_grad():
    total_result = transport_simulation(model, time_vector, state_all)


if(True):
    plot_image(total_result[100,0,:,:],filename = f'../results/carbon/full/test_flux.png')
    plot_image(total_result[100,1,:,:],filename = f'../results/carbon/full/test_concent_tt.png')
    plot_image(total_result[100,2,:,:],filename = f'../results/carbon/full/test_u.png')
    plot_image(total_result[100,3,:,:],filename = f'../results/carbon/full/test_v.png')

# exit()

"""
以下为数据同化部分
"""

# 获得同化的初始场
flux = 0.0*state0[0,:,:]
flux.requires_grad = True

# flux_all = flux.repeat([len(time_vector),1,1])
# flux_all.requires_grad = True
mask = torch.ones(u_all.shape)


c_all.requires_grad = False
u_all.requires_grad = False
v_all.requires_grad = False

def construct_state_flux_all(flux_all, mask, c_all, u_all, v_all):
    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all]).permute([1,0,2,3])
    """
    print(flux_mask.requires_grad)
    print(state_all.requires_grad)

    print(f"state_all:{state_all.shape}")
    """
    return state_all


def construct_state_flux_one(flux_one, mask, c_all, u_all, v_all):

    flux_all = flux_one.repeat([len(time_vector),1,1])
    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all]).permute([1,0,2,3])
    """
    print(flux_mask.requires_grad)
    print(state_all.requires_grad)

    print(f"state_all:{state_all.shape}")
    """
    return state_all


# state_all = construct_state_flux_all(flux, mask, c_all, u_all, v_all)
state_all = construct_state_flux_one(flux, mask, c_all, u_all, v_all)



# 开始梯度更新
optimizer = Adam([flux], lr=0.002)
criterion = torch.nn.MSELoss()

for iteration in range(300):
    time_start = time.time()

    state_full_pred = transport_simulation(model, time_vector, state_all)
    if(iteration % 10 ==0):
        plot_image(state_full_pred[100,1,:,:].detach(),filename = f'../results/carbon/full/test_concent_{iteration}.png')
        plot_image(state_full_pred[100,0,:,:].detach(),filename = f'../results/carbon/full/test_fulltime_flux_{iteration}_time100.png')
        plot_image(state_full_pred[50,0,:,:].detach(),filename = f'../results/carbon/full/test_fulltime_flux_{iteration}_time050.png')
        plot_image(state_full_pred[0,0,:,:].detach(),filename = f'../results/carbon/full/test_fulltime_flux_{iteration}_time000.png')

    loss = criterion(state_full_pred[:, 1:2, :, :],
                     total_result[:, 1:2, :, :])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

    """
    prepare for the next iteration, only update the flux. 
    由 可学习参数, 构造state
    """
    # state_all =  construct_state_flux_all(flux_all, mask, c_all, u_all, v_all)
    state_all = construct_state_flux_one(flux, mask, c_all, u_all, v_all)

