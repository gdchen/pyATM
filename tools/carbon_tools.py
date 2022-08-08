
import torch
from dynamic_model.initialize_tools import gkern



def plot_image(data, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(data[:, :], aspect='auto',
                   cmap='jet', animated=True)
    fig.colorbar(im)
    plt.savefig(filename, bbox_inches='tight')



def construct_state_flux_all(flux_all, mask, c_all, u_all, v_all):
    """
    对于变量进行拼接
    """
    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all]).permute([1,0,2,3])
    return state_all



def construct_state_flux_one(flux_one, mask, c_all, u_all, v_all):
    """
    通量不随时间变化
    """
    flux_all = flux_one.repeat([len(u_all),1,1])
    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all]).permute([1,0,2,3])
    """
    print(flux_mask.requires_grad)
    print(state_all.requires_grad)

    print(f"state_all:{state_all.shape}")
    """
    return state_all


def construct_state_flux_one_3d(flux_one, mask, c_all, u_all, v_all,w_all):
    """
    通量不随时间变化
    """
    flux_all = flux_one.repeat([len(u_all),1,1,1])
    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all,w_all]).permute([1,0,2,3,4])
    return state_all



def construct_state_flux_seven(flux_list, mask, c_all, u_all, v_all):

    """
    通量被划分为7份，随日期变化
    """

    len_each = len(u_all)//7 

    flux1,flux2,flux3,flux4,flux5,flux6,flux7 = flux_list
    flux_all = torch.vstack([flux1.repeat([len_each,1,1]),
                            flux2.repeat([len_each,1,1]),
                            flux3.repeat([len_each,1,1]),
                            flux4.repeat([len_each,1,1]),
                            flux5.repeat([len_each,1,1]),
                            flux6.repeat([len_each,1,1]),
                            flux7.repeat([len_each,1,1])])

    flux_mask = mask * flux_all 
    state_all = torch.stack([flux_mask, c_all, u_all, v_all]).permute([1,0,2,3])
    """
    print(flux_mask.requires_grad)
    print(state_all.requires_grad)

    print(f"state_all:{state_all.shape}")
    """
    return state_all



"""
以下代码为人工合成case
"""


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