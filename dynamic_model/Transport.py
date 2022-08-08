
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
from .differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
from .differ_module import Order2_Diff1_Unstructure,Order2_Diff2_Unstructure
from .differ_module import Order2_Diff1_Unstructure_Perioid_Upwind,Order2_Diff1_Unstructure_Upwind
from .initialize_tools import gkern
from .integral_module import rk4_step,rk2_step,rk1_step

from .ERA5_v2 import  filter_latitude, filter_longitude


# class Basic_Transport_Model(nn.Module):

#     def __init__(self, grid_info):
#         super(Basic_Transport_Model, self).__init__()
#         self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y = grid_info

#         self.nx = len(self.vector_x)
#         self.ny = len(self.vector_y)

#         self.diff1_x = Order2_Diff1_Unstructure_Period(
#             self.vector_x, total_dim=2, diff_dim=1)
        
#         self.diff2_x = Order2_Diff2_Unstructure_Period(
#             self.vector_x, total_dim=2, diff_dim=1)

#         self.diff1_y = Order2_Diff1_Unstructure_Period(
#             self.vector_y, total_dim=2, diff_dim=2)
#         self.diff2_y = Order2_Diff2_Unstructure_Period(
#             self.vector_y, total_dim=2, diff_dim=2)


#     def forward(self, t, state):
#         """
#         需要在不同时刻, 带入不同的速度
#         """
#         flux, concent, u, v = state
#         flux, concent, u, v = flux.unsqueeze(0),concent.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0)

#         # 定义边界层的厚度

#         # 定义边界层内的粘性系数

#         pbl_viscosity = 0.01
        
#         dconcent = flux - 1.0*self.diff1_x(concent*u) - 1.0*self.diff1_y(concent*v)
#         + pbl_viscosity *(self.diff2_x(concent) + self.diff2_y(concent))
        
#         dconcent = dconcent.squeeze()

#         dflux = torch.zeros(dconcent.shape)
#         du = 0.01*torch.randn(dconcent.shape)
#         dv = 0.01*torch.randn(dconcent.shape)
#         result = torch.stack([dflux, dconcent, du ,dv])
#         return result




class ERA5_transport_Model(nn.Module):
    def __init__(self, grid_info):

        super(ERA5_transport_Model, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y, self.map_factor = grid_info

        self.map_factor = torch.tensor(self.map_factor)
        # self.map_factor = torch.ones(self.map_factor.shape)

        print(f"map_factor:{torch.mean(self.map_factor, dim = (0))}")
        self.diff1_x = Order2_Diff1_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff1_y = Order2_Diff1_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

        self.diff2_x = Order2_Diff2_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

        # self.m2_diff_1_m = (self.map_factor**2 *
        #                     self.diff1_y(1.0/self.map_factor.unsqueeze(0))).squeeze()

        # self.f = torch.tensor(
        #     2*7.0e-5*np.sin(2*np.pi/360.0*self.grid_y/self.dy))

    def forward(self, t, state):
        flux, concent, u, v = state
        flux, concent, u, v = flux.unsqueeze(0),concent.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0)
        
        dconcent = flux - u*self.map_factor*1.0*self.diff1_x(concent) - v*1.0*self.diff1_y(concent)
        + 1.0 * (self.map_factor**2*self.diff2_x(concent) +
                      self.diff2_y(concent))
        
        dconcent = dconcent.squeeze()
        dflux = torch.zeros(dconcent.shape)
        du = 0.0*torch.randn(dconcent.shape)
        dv = 0.0*torch.randn(dconcent.shape)
        result = torch.stack([dflux, dconcent, du ,dv])
        return result





# class ERA5_transport_Model_3D(nn.Module):
class CTM_Model_3D(nn.Module):
    """
    没有加入边界层混合的程序
    """
    def __init__(self, grid_info,dim ):

        super(CTM_Model_3D, self).__init__()
        self.vector_x, self.vector_y, self.vector_z,\
            self.map_factor = grid_info
        
        self.dim = dim

        self.map_factor = torch.tensor(self.map_factor)


        self.diff1_x_upwind =  Order2_Diff1_Unstructure_Perioid_Upwind(
            self.vector_x, total_dim=3, diff_dim=1)

        self.diff1_y_upwind =   Order2_Diff1_Unstructure_Upwind(
            self.vector_y, total_dim=3, diff_dim=2)

        # self.diff1_x_upwind =   Order2_Diff1_Unstructure_Period(
        #             self.vector_x, total_dim=3, diff_dim=1)

        # self.diff1_y_upwind =   Order2_Diff1_Unstructure(
        #             self.vector_y, total_dim=3, diff_dim=2)


        if(self.dim ==3):
            self.diff1_z_upwind =   Order2_Diff1_Unstructure_Upwind(
                self.vector_z, total_dim=3, diff_dim=3)


    def forward(self, t, state):
        flux, concent, u, v, w = state
        flux, concent, u, v ,w  = \
            flux.unsqueeze(0),concent.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0), w.unsqueeze(0)
        u = torch.clip(u, -40, 40)
        v = torch.clip(v, -30, 30)
        w = torch.clip(w, -0.3, 0.3)
        
        if(self.dim ==3):
            """
            将风速clip在 cfl条件之内, 后续需要具体看
            """
            w_grad = self.diff1_z_upwind(concent, -w)
            w_grad[:,:,:,-1] = 0.0
            w_grad[:,:,:,0] = 0.0
            dconcent = flux - u*self.map_factor*1.0*self.diff1_x_upwind(concent, u) - v*1.0*self.diff1_y_upwind(concent,v) - (-w)*w_grad*1.0

            # dconcent = flux - u*self.map_factor*1.0*self.diff1_x_upwind(concent) - v*1.0*self.diff1_y_upwind(concent) - (-w)*w_grad*1.0

        if(self.dim ==2):
            dconcent = flux - u*self.map_factor*1.0*self.diff1_x_upwind(concent, u) - v*1.0*self.diff1_y_upwind(concent,v) 

            # dconcent = flux - u*self.map_factor*1.0*self.diff1_x_upwind(concent) - v*1.0*self.diff1_y_upwind(concent) 

        
        device = state.device
        dconcent = dconcent.squeeze().to(device)
        dflux = torch.zeros(dconcent.shape).to(device)
        du = 0.0*torch.randn(dconcent.shape).to(device)
        dv = 0.0*torch.randn(dconcent.shape).to(device)
        dw = 0.0*torch.randn(dconcent.shape).to(device)
        result = torch.stack([dflux, dconcent, du ,dv, dw])
        return result




def transport_simulation(model, time_vector, state_all):
    print("#"*20, "start transport simulation")
    print(state_all.shape)

    total_result = torch.empty(state_all.shape)
    total_result[0, ::] = state_all[0, ::]

    state = state_all[0,0:4,:,:]

    for index, time in enumerate(time_vector[1::]):
        dt = time_vector[index + 1] - time_vector[index]
        """
        f, u, v 从已知信息中读入
        """
        state[0:1,:,:] = state_all[index,0:1,:,:]
        state[2:4,:,:] = state_all[index,2:4,:,:]

        state = state + \
            rk4_step(model, time, dt, state)
        total_result[index + 1, :,:,:] = state
    return total_result




class SP_Filter(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, half_padding=1):
        super(SP_Filter, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")

        weights = torch.tensor(
            [[1.0/16.0,  2.0/16.0,  1.0/16.0], [2.0/16.0,  4.0/16.0,  2.0/16.0], [1.0/16.0,  2.0/16.0,  1.0/16.0]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        original_shape = u.shape
        u = u.squeeze(0)
        len_x, len_y = list(u.shape)[0], list(u.shape)[1]

        left_padder = u[:, len_y-self.half_padding: len_y]
        right_padder = u[:, 0: self.half_padding]
        u_pad = torch.cat([right_padder, u, left_padder],
                          dim=1).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = u_pad_forward[:, :, :, self.half_padding: len_y +
                                self.half_padding]

        return result.reshape(original_shape)






class SP_Filter_3D_temp(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, half_padding=1, channel_size = 1):
        super(SP_Filter_3D, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        # self.conv_layer = nn.Conv2d(
        #     1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")

        self.conv_layer = nn.Conv3d(
            channel_size, channel_size, (self.kernel_size, self.kernel_size, 1), padding=(1, 1, 0), padding_mode="circular")

        weights = torch.tensor(
            [[[1.0/16.0,  2.0/16.0,  1.0/16.0], [2.0/16.0,  4.0/16.0,  2.0/16.0], [1.0/16.0,  2.0/16.0,  1.0/16.0]]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size, 1)

        # weights = torch.tensor(
        #     [[[0.5/16.0,  1.0/16.0,  0.5/16.0], [0.5/16.0,  10.0/16.0,  0.5/16.0], [0.5/16.0,  1.0/16.0,  0.5/16.0]]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size, 1)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        original_shape = u.shape
        u = u.squeeze(0)
        len_x, len_y = list(u.shape)[0], list(u.shape)[1]

        # padding 一个值就行
        # 在 y 方向pad
        right_padder = u[:, len_y-self.half_padding: len_y]
        left_padder = u[:, 0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=2).unsqueeze(0).unsqueeze(0)
        # 求导

        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut  时间, x, y, z,
        result = u_pad_forward[:, :, :, self.half_padding: len_y +
                                self.half_padding]

        return result.reshape(original_shape)






class SP_Filter_3D(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, half_padding=1, channel_size = 1):
        super(SP_Filter_3D, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        # self.conv_layer = nn.Conv2d(
        #     1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")

        self.conv_layer = nn.Conv3d(
            channel_size, channel_size, (self.kernel_size, self.kernel_size, 1), padding=(1, 1, 0), padding_mode="circular")

        weights = torch.tensor(
            [[[1.0/16.0,  2.0/16.0,  1.0/16.0], [2.0/16.0,  4.0/16.0,  2.0/16.0], [1.0/16.0,  2.0/16.0,  1.0/16.0]]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size, 1)

        # weights = torch.tensor(
        #     [[[0.5/16.0,  1.0/16.0,  0.5/16.0], [0.5/16.0,  10.0/16.0,  0.5/16.0], [0.5/16.0,  1.0/16.0,  0.5/16.0]]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size, 1)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False


    def forward(self, u):
        u = u.unsqueeze(0)
        original_shape = u.shape
        # print("original shape:{original_shape}")
        len_x, len_y = list(u.shape)[2], list(u.shape)[3]
        

        # padding 一个值就行
        # 在 y 方向pad
        # print(f"u : {u.shape}")
        right_padder = u[:,:, len_y-self.half_padding: len_y]
        left_padder =  u[:,:, 0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=2)
        # print(f"u_pad : {u_pad.shape}")
        # 求导
        
        u_pad_forward = torch.empty(u_pad.shape)
        
        for i in range(u_pad_forward.shape[1] ):
            u_pad_forward[0:1,i:(i+1),:,:,:] = self.conv_layer(u_pad[0:1,i:(i+1),:,:,:])
            
        # u_pad_forward = self.conv_layer(u_pad)
        # print(f"u_pad_forward : {u_pad_forward.shape}")
        # 对 pad后的区域 cut  时间, x, y, z,
        result = u_pad_forward[:, :, self.half_padding: len_x +
                                self.half_padding,:,:]
        # print(f"result shape:{result.shape}")

        return result.reshape(original_shape).squeeze(0)


# def era5_transport_simulation(model, time_vector, state_all):

#     total_result = torch.empty(state_all.shape)
#     total_result[0, ::] = state_all[0, ::]

#     state = state_all[0,0:4,:,:]

#     filter  = SP_Filter()
#     for index, time in enumerate(time_vector[1::]):
#         dt = time_vector[index + 1] - time_vector[index]
#         """
#         f, u, v 从已知信息中读入
#         """
#         state[0:1,:,:] = state_all[index,0:1,:,:]
#         state[2:4,:,:] = state_all[index,2:4,:,:]

#         state = state + \
#             rk4_step(model, time, dt, state)

        
#         state[1:2,:,:]  = 0.9*state[1:2,:,:] + 0.1*filter.forward(state[1:2,:,:])
#         state[1, :, 0] = torch.mean(state[1, :, 1], dim=(0))
#         state[1, :, -1] = torch.mean(state[1, :, -2], dim=(0))

#         total_result[index + 1, :,:,:] = state
#     return total_result




# def era5_transport_simulation_3d(model, time_vector, state_all):
#     total_result = torch.empty(state_all.shape)
#     total_result[0, ::] = state_all[0, ::]

#     state = state_all[0,0:5,::]
    
#     filter  = SP_Filter_3D()
#     for index, time in enumerate(time_vector[1::]):
#         dt = time_vector[index + 1] - time_vector[index]
#         """
#         f, u, v, w 从已知信息中读入
#         """
#         state[0:1,::] = state_all[index,0:1,::]
#         state[2:5,::] = state_all[index,2:5,::]

#         state = state + \
#             rk2_step(model, time, dt, state)
        
#         state[1:2,::]  = 0.998*state[1:2,::] \
#          + 0.002*filter.forward(state[1:2,::])

    
#         # 边界处的值用平均值代替， index  1 为浓度 
#         upper_mean =  torch.mean(state[1, :, 0:2,:], dim=(0,1))
#         state[1, :, 0:2,:] = upper_mean

#         lower_mean =  torch.mean(state[1, :, -2::,:], dim=(0,1))
#         state[1, :, -2::,:] = lower_mean

#         total_result[index + 1, :,:,:,:] = state

#     return total_result



def ctm_simulation_3d_mixing(model, time_vector, state_all, pbl_top,
                                        vertical_mapping, weight_z, if_mixing = False):
    # print(f"if_mixing:{if_mixing}")
    device = state_all.device
    total_result = torch.empty(state_all.shape, device = device)
    total_result[0, ::] = state_all[0, ::]
    state = state_all[0,0:5,::]
    pbl_top = pbl_top.to(torch.int)
    
    # filter = SP_Filter_3D()
    # 从mol m-2 s-1 到 体积浓度/s-1 的转换关系
    bottom_flux_factor = 1.0/(weight_z[0]*10.39*1000/29e-3)
    
    for index, time in enumerate(time_vector[1::]):
        dt = time_vector[index + 1] - time_vector[index]
        """
        f, u, v, w 从已知信息中读入
        """
        state[0:1,::] = state_all[index,0:1,::] * bottom_flux_factor
        
        # print(torch.max(state[0:1,::]), torch.min(state[0:1,::]))
        state[2:5,::] = state_all[index,2:5,::]

        # 采用rk1 和 rk4 效果相同 
        state = state + rk4_step(model, time, dt, state)
        
        # state[1:2,::]  = 0.95*state[1:2,::] \
        #           + 0.05*filter.forward(state[1:2,::])
            
        # 仿照geos-chem 边界处的值用平均值代替， index  1 为浓度 
        upper_mean =  torch.mean(state[1, :, 0:2,:], dim=(0,1))
        state[1, :, 0:2,:] = upper_mean

        lower_mean =  torch.mean(state[1, :, -2::,:], dim=(0,1))
        state[1, :, -2::,:] = lower_mean
        
        """
        是否开启边界层的混合
        """
        if(if_mixing & (index%1 == 0)):
        # if(if_mixing ):
            pbl_top_current = pbl_top[index,0,:,:]
            state[1,:,:,:] = turbday_model(pbl_top_current, state[1,:,:,:], 
                                           vertical_mapping, weight_z)

            # state[1,:,:,:] = turbday_model_naive(pbl_top_current, state[1,:,:,:], 
            #                                vertical_mapping, weight_z)
                                           
        

        state[0:1,::] = state[0:1,::]/bottom_flux_factor
        total_result[index + 1, :,:,:,:] = state
    return total_result

#%%

def turbday_model_naive(pbl_top_int, concentration_tensor,vertical_mapping,weight_z):
    
    """
    采用简单的 ij for 的格式实现tubday 格式的比较耗时, 默认不使用，代码留着
    """
    shape0, shape1 = concentration_tensor.shape[0], concentration_tensor.shape[1]
    
    for i in range(shape0):
        for j in range(shape1):
            pbl_top_current_index = pbl_top_int[i,j]
            #如果不是全部高度层，此处需要有转换
            avg_height_index = vertical_mapping["backward"][pbl_top_current_index.item()]
            
            # print(concentration_tensor[ i, j,0:avg_height_index].shape, weight_z.shape)

            mean_value = concentration_tensor[ i, j,0:avg_height_index]* weight_z[0:avg_height_index]/sum(weight_z[0:avg_height_index])
            concentration_tensor[i, j, 0:avg_height_index] = mean_value
    return concentration_tensor
    
    

def turbday_model(pbl_top_int, concentration_tensor,vertical_mapping, weight_z):
    """
    按pbl高度来做， 
    step1. 找对对应的index， 计算平均值并赋值
    """
    shape0, shape1 = concentration_tensor.shape[0],concentration_tensor.shape[1]
    first_dim = shape0 * shape1
    concentration_tensor_reshape =  torch.reshape(concentration_tensor, [first_dim, -1])
    # max_height = torch.max(pbl_top_int).to(torch.int)
    for height in list(vertical_mapping["forward"].values()):
        pbl0_tf = ((pbl_top_int == height) == 1)
        pbl0_tf_flatten = pbl0_tf.flatten()    
        pbl0_tf_flatten_index = torch.where(pbl0_tf_flatten)[0]
        # print(f"height:{height}, number:{ torch.sum(pbl0_tf_flatten_index) }")
        #如果不是全部高度层，此处需要有转换
        avg_height_index = vertical_mapping["backward"][height] + 1
        # temp_old = 1.0*torch.mean(concentration_tensor_reshape[pbl0_tf_flatten_index, 0:avg_height_index],axis = 1)
        
        temp = torch.sum(concentration_tensor_reshape[pbl0_tf_flatten_index, 0:avg_height_index]*weight_z[0:avg_height_index],axis = 1)/torch.sum(weight_z[0:avg_height_index])
        #concentration_tensor_reshape[pbl0_tf_flatten_index, 0:avg_height_index] = 1.0*torch.mean(concentration_tensor_reshape[pbl0_tf_flatten_index, 0:avg_height_index],axis = 1).unsqueeze(-1)
        concentration_tensor_reshape[pbl0_tf_flatten_index, 0:avg_height_index] = 1.0*temp.unsqueeze(-1)
    tensor_tf_unreshape = torch.reshape(concentration_tensor_reshape, [shape0, shape1 ,-1 ])
    return tensor_tf_unreshape

