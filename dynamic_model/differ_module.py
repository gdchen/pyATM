#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen
"""
# %%
import torch
import torch.nn as nn


class Order2_Diff1_Old(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff1_Old, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv1d(
            1, 1, self.kernel_size, padding=1, padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [-0.5,  0.0,  0.5], dtype=torch.float32).view(1, 1, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        """
        write the dimension of the forward
        """
        original_shape = u.shape
        u = u.squeeze(0)

        original_len = list(u.shape)[0]
        left_padder = u[original_len-self.half_padding: original_len]
        right_padder = u[0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=0).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, self.half_padding: original_len +
                                self.half_padding]/self.dx)

        return result.reshape(original_shape)


class Order2_Diff2_Old(nn.Module):
    """
    second order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff2_Old, self).__init__()

        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv1d(
            1, 1, self.kernel_size, padding=1, padding_mode="circular")
        self.dx = dx
        # with torch.no_grad():
        weights = torch.tensor(
            [1.0,  -2.0,  1.0], dtype=torch.float32).view(1, 1, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        # 需要pad
        original_shape = u.shape

        u = u.squeeze(0)
        original_len = list(u.shape)[0]
        left_padder = u[original_len-self.half_padding: original_len]
        right_padder = u[0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=0).unsqueeze(0).unsqueeze(0)

        # 求导
        u_pad = self.conv_layer(u_pad)

        # 对 pad后的区域 cut
        result = (u_pad[:, :, self.half_padding: original_len +
                        self.half_padding]/self.dx/self.dx)
        return result.reshape(original_shape)


class Order2_Diff1(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff1, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv1d(
            1, 1, self.kernel_size, padding=1, padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [-0.5,  0.0,  0.5], dtype=torch.float32).view(1, 1, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        """
        write the dimension of the forward
        input: 3 dimensional vecter batch_size, variable_size, dx
        """
        x_len = list(u.shape)[2]

        left_padder = u[:, :, x_len-self.half_padding: x_len]
        right_padder = u[:, :, 0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=2)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, self.half_padding: x_len +
                                self.half_padding]/self.dx)
        return result


class Order2_Diff2(nn.Module):
    """
    second order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff2, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv1d(
            1, 1, self.kernel_size, padding=1, padding_mode="circular")
        self.dx = dx
        # with torch.no_grad():
        weights = torch.tensor(
            [1.0,  -2.0,  1.0], dtype=torch.float32).view(1, 1, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        """
        write the dimension of the forward
        input: 3 dimensional vecter batch_size, variable_size, dx
        """
        # 需要pad
        x_len = list(u.shape)[2]
        left_padder = u[:, :, x_len-self.half_padding: x_len]
        right_padder = u[:, :, 0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=2)
        # 求导
        u_pad = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad[:, :, self.half_padding: x_len +
                        self.half_padding]/self.dx/self.dx)
        return result
# %%


class Order2_Diff1_2D_X(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff1_2D_X, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [[0.0,  -0.5,  0.0], [0.0,  0.0,  0.0], [0.0,  0.5,  0.0]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        original_shape = u.shape
        u = u.squeeze(0)
        len_x, len_y = list(u.shape)[0], list(u.shape)[1]

        left_padder = u[len_x-self.half_padding: len_x]
        right_padder = u[0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=0).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, self.half_padding: len_x +
                                self.half_padding]/self.dx)

        return result.reshape(original_shape)


class Order2_Diff1_2D_Y(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff1_2D_Y, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, kernel_size=(self.kernel_size, self.kernel_size), padding=1, stride=1, padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float32).view(1, 1,  self.kernel_size, self.kernel_size)
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
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=1).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, :, self.half_padding: len_y +
                                self.half_padding]/self.dx)

        return result.reshape(original_shape)


class Order2_Diff2_2D_X(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff2_2D_X, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [[0.0,  1.0,  0.0], [0.0,  -2.0,  0.0], [0.0,  1.0,  0.0]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        original_shape = u.shape
        u = u.squeeze(0)
        len_x, len_y = list(u.shape)[0], list(u.shape)[1]

        left_padder = u[len_x-self.half_padding: len_x]
        right_padder = u[0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=0).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, self.half_padding: len_x +
                                self.half_padding]/self.dx/self.dx)

        return result.reshape(original_shape)


class Order2_Diff2_2D_Y(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff2_2D_Y, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [[0.0,  0.0,  0.0], [1.0,  -2.0,  1.0], [0.0,  0.0,  0.0]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
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
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=1).unsqueeze(0).unsqueeze(0)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, :, self.half_padding: len_y +
                                self.half_padding]/self.dx/self.dx)

        return result.reshape(original_shape)


"""
迎风格式从这里开始， 周期和非周期部分单独来做！！！

"""
class Order2_Diff1_Unstructure_Perioid_forward(nn.Module):
    """
    2阶 正向的风的梯度，周期条件，用于后续构造迎风格式
    m-2,m-1，m
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1, dtype = "float32"):
        super(Order2_Diff1_Unstructure_Perioid_forward, self).__init__()

        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim
        device = z_vector.device
        
        # print("#"*20)
        
        dz = z_vector[1] - z_vector[0]
        # print(f"z_vector device:{z_vector.device}")
        dz_m1_list = torch.ones(z_vector.shape, device = device)*dz
        dz_m2_list = torch.ones(z_vector.shape, device = device)*dz
        self.weights_all = torch.empty([self.len, 3], device = device)
        if( dtype == "float64"):
            self.weights_all = self.weights_all.to(torch.float64)
        # print("#"*20)
        for index in range(0, self.len):
            dz_m1 = dz_m1_list[index]
            dz_m2 = dz_m2_list[index]

            factor = - dz_m1 * (dz_m1 + dz_m2) / dz_m2

            weights = torch.tensor(
                    [-1.0*factor/(dz_m2 + dz_m1)**2,
                     1.0*factor/(dz_m1)**2,
                     1.0*factor/(dz_m2 + dz_m1)**2 - 1.0*factor/(dz_m1)**2, 
                     ],device = device)
            
            # print(weights)
            self.weights_all[index] = weights
        
        "无需更新"
        self.weights_all.requires_grad = False


    def forward(self, u ): 

        """
        schema: batch, variable, x, y, z
        """
        u_m1 = torch.roll(u, 1, self.diff_dim)
        u_m2 = torch.roll(u, 2, self.diff_dim)
        u_pad = torch.stack([u_m2, u_m1, u], dim=len(u.shape))
        
        
        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):

            if(self.diff_dim == 1):
                # print(u_pad.dtype, self.weights_all.dtype)
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)

        return result

    



class Order2_Diff1_Unstructure_Perioid_backward(nn.Module):
    """
    2阶 负向的风的梯度，周期条件，用于后续构造迎风格式
    p, p1，p2
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1, dtype = "float32"):
        super(Order2_Diff1_Unstructure_Perioid_backward, self).__init__()

        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim

        dz = z_vector[1] - z_vector[0]
        
        device = z_vector.device
        
        dz_p1_list = torch.ones(z_vector.shape, device = device)*dz
        dz_p2_list = torch.ones(z_vector.shape, device = device)*dz
        self.weights_all = torch.empty([self.len, 3], device = device)
        
        if( dtype == "float64"):
            self.weights_all = self.weights_all.to(torch.float64)
        


        for index in range(0, self.len):
            dz_p1 = dz_p1_list[index]
            dz_p2 = dz_p2_list[index]

            factor = dz_p1 * (dz_p1 + dz_p2) / dz_p2
            weights = torch.tensor(
                    [1.0*factor/(dz_p1 + dz_p2)**2 - 1.0*factor/(dz_p1)**2,
                     1.0*factor/(dz_p1)**2,
                     -1.0*factor/(dz_p1 + dz_p2)**2, 
                     ], device = device)
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False


    def forward(self, u ): 

        """
        schema: batch, variable, x, y, z
        """
        u_p1 = torch.roll(u, -1, self.diff_dim)
        u_p2 = torch.roll(u, -2, self.diff_dim)
        u_pad = torch.stack([u, u_p1, u_p2], dim=len(u.shape))
        
        
        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):

            if(self.diff_dim == 1):
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)

        return result
    

class Order2_Diff1_Unstructure_Perioid_Upwind(nn.Module):
    """
    2阶 负向的风的梯度，周期条件，用于后续构造迎风格式
    p, p1，p2
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1,dtype = "float32"):
        super(Order2_Diff1_Unstructure_Perioid_Upwind, self).__init__()
        
        self.order2_Diff1_Unstructure_Perioid_forward = Order2_Diff1_Unstructure_Perioid_forward(z_vector, total_dim, diff_dim,dtype)
        self.order2_Diff1_Unstructure_Perioid_backward = Order2_Diff1_Unstructure_Perioid_backward(z_vector, total_dim, diff_dim,dtype)
        
    
    def forward(self, u, u_sign):
        u_forward = self.order2_Diff1_Unstructure_Perioid_forward(u)
        u_backward = self.order2_Diff1_Unstructure_Perioid_backward(u)
        
        return (u_sign >=0.0) * u_forward + (u_sign<0.0)* u_backward
        
"""
迎风格式从这里开始， 非周期部分

"""
class Order2_Diff1_Unstructure_forward(nn.Module):
    """
    2阶 正向的风的梯度，非周期条件，用于前向构造迎风格式
    m-2,m-1，m
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1,dtype = "float32"):
        super(Order2_Diff1_Unstructure_forward, self).__init__()

        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim

        dz_shift1_list = z_vector[1::] - z_vector[0:-1]
        dz_shift2_list = z_vector[2::] - z_vector[0:-2]
        
        device = z_vector.device
        dz_m1_list = torch.cat([torch.tensor([0.0], device = device ), dz_shift1_list])
        dz_m2_list = torch.cat([torch.tensor([0.0,0.0],device = device ), dz_shift2_list])
        
        
        self.weights_all = torch.empty([self.len, 3], device = device)
        
        if( dtype == "float64"):
            self.weights_all = self.weights_all.to(torch.float64)
        
        # 降到一阶
        if(True):
            self.weights_all[0] = torch.tensor(
                [0.0, 0.0, 0.0], device = device)
            
            self.weights_all[1] = torch.tensor(
                [0.0, -1.0/dz_m1_list[1], 1.0/dz_m1_list[1]], device = device)

        for index in range(2, self.len):
            dz_m1 = dz_m1_list[index]
            dz_m2 = dz_m2_list[index]
            
            dz_m2 = dz_m2 - dz_m1

            factor = -dz_m1 * (dz_m1 + dz_m2) / dz_m2

            weights = torch.tensor(
                    [-1.0*factor/(dz_m2 + dz_m1)**2,
                     1.0*factor/(dz_m1)**2,
                     1.0*factor/(dz_m2 + dz_m1)**2 - 1.0*factor/(dz_m1)**2, 
                     ], device = device)
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False


    def forward(self, u ): 

        """
        schema: batch, variable, x, y, z
        """
        u_m1 = torch.roll(u, 1, self.diff_dim)
        u_m2 = torch.roll(u, 2, self.diff_dim)
        u_pad = torch.stack([u_m2, u_m1, u], dim=len(u.shape))
        
        
        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):

            if(self.diff_dim == 1):
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)

        return result
    


class Order2_Diff1_Unstructure_backward(nn.Module):
    """
    2阶 正向的风的梯度，非周期条件，用于后向构造迎风格式
    m-2,m-1，m
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1, dtype = "float32"):
        super(Order2_Diff1_Unstructure_backward, self).__init__()

        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim

        dz_shift1_list = z_vector[1::] - z_vector[0:-1]
        dz_shift2_list = z_vector[2::] - z_vector[0:-2]
        
        device = z_vector.device
        dz_p1_list = torch.cat([dz_shift1_list,torch.tensor([0.0], device = device)])
        dz_p2_list = torch.cat([dz_shift2_list,torch.tensor([0.0,0.0], device = device) ])
        
        
        self.weights_all = torch.empty([self.len, 3], device = device)
        
        if( dtype == "float64"):
            self.weights_all = self.weights_all.to(torch.float64)
        
        # 降到一阶
        if(True):
            self.weights_all[-1] = torch.tensor(
                [0.0, 0.0, 0.0], device = device)
            
            self.weights_all[-2] = torch.tensor(
                [-1.0/dz_p1_list[-2], 1.0/dz_p1_list[-2], 0.0], device = device)

        for index in range(0, self.len-2):
            dz_p1 = dz_p1_list[index]
            dz_p2 = dz_p2_list[index]
            
            dz_p2 = dz_p2 - dz_p1

            factor = dz_p1 * (dz_p1 + dz_p2) / dz_p2
            weights = torch.tensor(
                    [1.0*factor/(dz_p1 + dz_p2)**2 - 1.0*factor/(dz_p1)**2,
                     1.0*factor/(dz_p1)**2,
                     -1.0*factor/(dz_p1 + dz_p2)**2, 
                     ], device = device)
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False


    def forward(self, u ): 

        """
        schema: batch, variable, x, y, z
        """
        u_p1 = torch.roll(u, -1, self.diff_dim)
        u_p2 = torch.roll(u, -2, self.diff_dim)
        u_pad = torch.stack([u, u_p1, u_p2], dim=len(u.shape))
        
        
        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):

            if(self.diff_dim == 1):
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)

        return result




class Order2_Diff1_Unstructure_Upwind(nn.Module):
    """
    2阶 负向的风的梯度，周期条件，用于后续构造迎风格式
    p, p1，p2
    """
    def __init__(self, z_vector, total_dim=1, diff_dim=1, dtype = "float32"):
        super(Order2_Diff1_Unstructure_Upwind, self).__init__()
        
        self.order2_Diff1_Unstructure_forward = Order2_Diff1_Unstructure_forward(z_vector, total_dim, diff_dim,dtype)
        self.order2_Diff1_Unstructure_backward = Order2_Diff1_Unstructure_backward(z_vector, total_dim, diff_dim,dtype)
        
    
    def forward(self, u, u_sign):
        u_forward = self.order2_Diff1_Unstructure_forward(u)
        u_backward = self.order2_Diff1_Unstructure_backward(u)
        
        return (u_sign >=0.0) * u_forward + (u_sign<0.0)* u_backward
        
    
    

class Order2_Diff1_Unstructure(nn.Module):
    def __init__(self, z_vector, total_dim=1, diff_dim=1,dtype = "float32"):
        super(Order2_Diff1_Unstructure, self).__init__()
        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim
        dz_list = z_vector[1::] - z_vector[0:-1]
        # dz_m 0号网格没有minus, 最后1个网格没有 plus
        device = z_vector.device
        dz_m_list = torch.cat([torch.tensor([dz_list[1]], device = device), dz_list])
        dz_p_list = torch.cat([dz_list, torch.tensor([dz_list[self.len-3]], device = device)])
        self.weights_all = torch.empty([self.len, 3],device = device)
        
        if(dtype == "float64"):
            dz_m_list = dz_m_list.to(torch.float64)
            dz_p_list = dz_p_list.to(torch.float64)
            self.weights_all = self.weights_all.to(torch.float64)
    
        # 降到一阶
        if(True):
            self.weights_all[0] = torch.tensor(
                [0.0, -1.0/dz_p_list[0], 1.0/dz_p_list[0]], device = device)
            self.weights_all[self.len-1] = torch.tensor(
                [-1.0 / dz_m_list[self.len - 1], 1.0 / dz_m_list[self.len - 1], 0.0],  device = device)

        if(False):
            index = 0
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            factor = dz_p * (dz_p + dz_m) / dz_m
            self.weights_all[index] = torch.tensor(
                [-1.0*factor/(dz_p + dz_m)**2, 1.0*factor/(dz_p + dz_m)**2 - 1.0*factor/(dz_p)**2, 1.0*factor/(dz_p)**2], device = device)

            index = self.len-1
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            factor = dz_m * (dz_m + dz_p) / dz_p
            self.weights_all[index] = torch.tensor(
                [-1.0*factor/(dz_m)**2, -1.0*factor/(dz_p + dz_m)**2 + 1.0*factor/(dz_m)**2, +1.0*factor/(dz_p + dz_m)**2], device = device)

        for index in range(1, self.len-1):
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            weights = torch.tensor(
                [-1.0 / dz_m / dz_m * (dz_m * dz_p) / (dz_m + dz_p),
                    (1.0 / dz_m / dz_m - 1.0 / dz_p / dz_p) *
                 (dz_m * dz_p) / (dz_m + dz_p),
                 1.0 / dz_p / dz_p * (dz_m * dz_p) / (dz_m + dz_p)], device = device)
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False

    def forward(self, u):
        """
        schema: batch, variable, x, y, z
        """

        u_m = torch.roll(u, 1, self.diff_dim)
        # u_m[..., 0] = u[..., 2]

        u_p = torch.roll(u, -1, self.diff_dim)
        # u_p[..., -1] = u[..., -3]

        # u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

        if (self.total_dim == 1):
            if (self.diff_dim == 1):
                u_m[:, 0] = u[:, 2]
                u_p[:, -1] = u[:, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):
            if (self.diff_dim == 1):
                u_m[:, 0, :] = u[:, 2, :]
                u_p[:, -1, :] = u[:, -3, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)
            if (self.diff_dim == 2):

                u_m[:, :, 0] = u[:, :, 2]
                u_p[:, :, -1] = u[:, :, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):

                u_m[:, 0, :, :] = u[:, 2, :, :]
                u_p[:, -1, :, :] = u[: -3, :, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 2):

                u_m[:, :, 0, :] = u[:, :, 2, :]
                u_p[:, :, -1, :] = u[:, :, -3, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                u_m[:, :, :, 0] = u[:, :, :, 2]
                u_p[:, :, :, -1] = u[:, :, :, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)

        return result.reshape(u.shape)


class Order2_Diff1_Unstructure_Period(nn.Module):
    """
    total_dim = [1,2,3]
    diff_dim = [1,2,3]
    中心格式 
    """

    def __init__(self, z_vector, total_dim=1, diff_dim=1, dtype = "float32"):
        super(Order2_Diff1_Unstructure_Period, self).__init__()
        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim

        dz = z_vector[1] - z_vector[0]
        dz_m_list = torch.ones(z_vector.shape)*dz
        dz_p_list = torch.ones(z_vector.shape)*dz
        self.weights_all = torch.empty([self.len, 3])
        
        if(dtype == "float64"):
            dz_m_list = dz_m_list.to(torch.float64)
            dz_p_list = dz_p_list.to(torch.float64)
            self.weights_all = self.weights_all.to(torch.float64)

        for index in range(0, self.len):
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            weights = torch.tensor(
                [-1.0 / dz_m / dz_m * (dz_m * dz_p) / (dz_m + dz_p),
                    (1.0 / dz_m / dz_m - 1.0 / dz_p / dz_p) *
                 (dz_m * dz_p) / (dz_m + dz_p),
                 1.0 / dz_p / dz_p * (dz_m * dz_p) / (dz_m + dz_p)])
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False

    def forward(self, u):
        """
        schema: batch, variable, x, y, z
        """
        u_m = torch.roll(u, 1, self.diff_dim)
        u_p = torch.roll(u, -1, self.diff_dim)
        u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)
        if(self.total_dim == 2):
            if(self.diff_dim == 1):
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)
        return result.reshape(u.shape)


class Order2_Diff2_Unstructure_Period(nn.Module):
    """
    total_dim = [1,2,3]
    diff_dim = [1,2,3]
    """

    def __init__(self, z_vector, total_dim=1, diff_dim=1,dtype = "float32"):
        super(Order2_Diff2_Unstructure_Period, self).__init__()
        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim
        
        device = z_vector.device
        dz = z_vector[1] - z_vector[0]
        dz_m_list = torch.ones(z_vector.shape, device = device)*dz
        dz_p_list = torch.ones(z_vector.shape, device = device )*dz
        self.weights_all = torch.empty([self.len, 3],device = device)
        
        if(dtype == "float64"):
            dz_m_list = dz_m_list.to(torch.float64)
            dz_p_list = dz_p_list.to(torch.float64)
            self.weights_all = self.weights_all.to(torch.float64)
    
    
        # 降到一阶
        for index in range(0, self.len):
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            weights = torch.tensor(
                [2.0 / dz_m / (dz_m + dz_p),
                 -2.0 / dz_m / (dz_m + dz_p) - 2.0 / dz_p / (dz_m + dz_p),
                 2.0 / dz_p / (dz_m + dz_p)], device = device)
            self.weights_all[index] = weights
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False

    def forward(self, u):
        """
        schema: batch, variable, x, y, z
        """

        u_m = torch.roll(u, 1, self.diff_dim)
        u_p = torch.roll(u, -1, self.diff_dim)
        u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

        if(self.total_dim == 1):
            if(self.diff_dim == 1):
                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):

            if(self.diff_dim == 1):
                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)

            if(self.diff_dim == 2):
                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):
                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if(self.diff_dim == 2):
                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)
        return result.reshape(u.shape)


class Order2_Diff2_Unstructure(nn.Module):

    """
    边界用内点来代替， -1, 0, 1. 需要用到 2 网格点，换到-1点。
    """

    def __init__(self, z_vector, total_dim=1, diff_dim=1):
        super(Order2_Diff2_Unstructure, self).__init__()
        self.len = z_vector.shape[0]
        self.total_dim = total_dim
        self.diff_dim = diff_dim

        dz_list = z_vector[1::] - z_vector[0:-1]
        # dz_m 0号网格没有minus, 最后1个网格没有 plus
        
        device = z_vector.device
        dz_m_list = torch.cat([torch.tensor([dz_list[1]], device =device), dz_list])
        dz_p_list = torch.cat([dz_list, torch.tensor([dz_list[self.len-3]], device =device)])

        self.weights_all = torch.empty([self.len, 3], device =device)

        index = 0
        dz_m = dz_m_list[index]
        dz_p = dz_p_list[index]
        self.weights_all[index] = torch.tensor(
            [2.0 / dz_m / (dz_m + dz_p), 2.0 / dz_p / (dz_m + dz_p), -2.0 / dz_p / dz_m], device =device)

        # self.weights_all[index] = torch.tensor(
        #     [2.0 / dz_p / dz_m - 2.0 / dz_p / (dz_m + dz_p), 2.0 / dz_p / (dz_m + dz_p), -2.0 / dz_p / dz_m])

        index = self.len - 1
        dz_m = dz_m_list[index]
        dz_p = dz_p_list[index]
        self.weights_all[index] = torch.tensor(
            [2.0 / dz_m / (dz_p), -2.0 / dz_m / (dz_m + dz_p), -2.0 / dz_p / (dz_m + dz_p)], device =device)

        # self.weights_all[self.len-1] = torch.tensor(
        #     [-1.0 / (dz_m_list[self.len - 1] ** 2), 1.0 / (dz_m_list[self.len - 1] ** 2), 0.0])

        for index in range(1, self.len-1):
            dz_m = dz_m_list[index]
            dz_p = dz_p_list[index]
            weights = torch.tensor(
                [2.0 / dz_m / (dz_m + dz_p),
                 -2.0 / dz_m / (dz_m + dz_p) - 2.0 / dz_p / (dz_m + dz_p),
                 2.0 / dz_p / (dz_m + dz_p)], device =device)
            self.weights_all[index] = weights
        "无需更新"
        self.weights_all.requires_grad = False
        # print(self.weights_all)

    def forward(self, u):
        """
        schema: batch, variable, x, y, z

        u [variable,x,y,z]
        variable, self.diff_dim ==0 , 等于 variable, self.diff_dim ==2
        """
        u_m = torch.roll(u, 1, self.diff_dim)
        # u_m[..., 0] = u[..., 2]

        u_p = torch.roll(u, -1, self.diff_dim)
        # u_p[..., -1] = u[..., -3]

        # u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

        # if(self.total_dim == 1):
        #     if(self.diff_dim == 1):
        #         result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        # if(self.total_dim == 2):

        #     if(self.diff_dim == 1):
        #         result = torch.einsum(
        #             "vxzt,xt->vxz", u_pad, self.weights_all)

        #     if(self.diff_dim == 2):
        #         result = torch.einsum(
        #             "vxzt,zt->vxz", u_pad, self.weights_all)

        # if(self.total_dim == 3):
        #     if (self.diff_dim == 1):
        #         result = torch.einsum("vxyzt,xt->vxyz",
        #                               u_pad, self.weights_all)
        #     if(self.diff_dim == 2):
        #         result = torch.einsum("vxyzt,yt->vxyz",
        #                               u_pad, self.weights_all)
        #     if (self.diff_dim == 3):
        #         result = torch.einsum("vxyzt,zt->vxyz",
        #                               u_pad, self.weights_all)
        if (self.total_dim == 1):
            if (self.diff_dim == 1):
                u_m[:, 0] = u[:, 2]
                u_p[:, -1] = u[:, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vzt,zt->vz", u_pad, self.weights_all)

        if(self.total_dim == 2):
            if (self.diff_dim == 1):
                u_m[:, 0, :] = u[:, 2, :]
                u_p[:, -1, :] = u[:, -3, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum(
                    "vxzt,xt->vxz", u_pad, self.weights_all)
            if (self.diff_dim == 2):

                u_m[:, :, 0] = u[:, :, 2]
                u_p[:, :, -1] = u[:, :, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum(
                    "vxzt,zt->vxz", u_pad, self.weights_all)

        if(self.total_dim == 3):
            if (self.diff_dim == 1):

                u_m[:, 0, :, :] = u[:, 2, :, :]
                u_p[:, -1, :, :] = u[: -3, :, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vxyzt,xt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 2):

                u_m[:, :, 0, :] = u[:, :, 2, :]
                u_p[:, :, -1, :] = u[:, :, -3, :]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))

                result = torch.einsum("vxyzt,yt->vxyz",
                                      u_pad, self.weights_all)
            if (self.diff_dim == 3):
                u_m[:, :, :, 0] = u[:, :, :, 2]
                u_p[:, :, :, -1] = u[:, :, :, -3]
                u_pad = torch.stack([u_m, u, u_p], dim=len(u.shape))
                result = torch.einsum("vxyzt,zt->vxyz",
                                      u_pad, self.weights_all)
        return result.reshape(u.shape)


def Laplace(x_vector, y_vector):
    """
    返回2维平面上的laplace 算子
    """
    return sc.dot(Dx, Gx) + sc.dot(Dy, Gy)


def main():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    dx = 0.01
    dy = 0.01
    x = torch.arange(0,  100*dx, dx, dtype=torch.float32)
    y = torch.arange(0,  100*dy, dy, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y)
    u = torch.sin(2*np.pi * grid_x)
    v = torch.sin(2*np.pi * grid_y)
    plt.imshow(u)

    order2_diff1_2d_x = Order2_Diff1_2D_X(dx=dx, half_padding=1)
    order2_diff1_2d_y = Order2_Diff1_2D_Y(dx=dy, half_padding=1)

    order2_diff2_2d_x = Order2_Diff2_2D_X(dx=dx, half_padding=1)
    order2_diff2_2d_y = Order2_Diff2_2D_Y(dx=dy, half_padding=1)

    plt.cla()
    plt.subplot(221)

    # equivalent but more general
    ax1 = plt.subplot(2, 2, 1)
    tt1 = order2_diff2_2d_y(v)
    plt.imshow(tt1)

    ax2 = plt.subplot(2, 2, 2)
    tt2 = order2_diff1_2d_y(v)
    plt.imshow(tt2)

    ax3 = plt.subplot(2, 2, 3)
    tt3 = order2_diff2_2d_x(u)
    plt.imshow(tt3)

    ax4 = plt.subplot(2, 2, 4)
    tt4 = order2_diff2_2d_y(v)
    plt.imshow(tt4)
    plt.savefig("differ_modeule.png")
    
    
    """
    周期迎风格式测试
    """
    plt.figure(2)
    order2_Diff1_Unstructure_forward  = Order2_Diff1_Unstructure_Perioid_forward( x, total_dim=1, diff_dim=1)
    order2_Diff1_Unstructure_backward  = Order2_Diff1_Unstructure_Perioid_backward( x, total_dim=1, diff_dim=1)
    order2_Diff1_Unstructure_upwind = Order2_Diff1_Unstructure_Perioid_Upwind( x, total_dim=1, diff_dim=1)

    u = torch.sin(2*np.pi*x).unsqueeze(0)
    u_diff1_forward  = order2_Diff1_Unstructure_forward(u)
    u_diff1_backward = order2_Diff1_Unstructure_backward(u)
    u_diff1_upwind = order2_Diff1_Unstructure_upwind(u,u)
    
    
    plt.plot(x, u.squeeze())
    plt.plot(x, u_diff1_forward.squeeze())
    plt.plot(x, u_diff1_backward.squeeze())
    plt.plot(x, u_diff1_upwind.squeeze())


    """
    非周期迎风格式测试
    """
    plt.figure(3)
    order2_Diff1_Unstructure_forward  = Order2_Diff1_Unstructure_forward( x, total_dim=1, diff_dim=1)
    order2_Diff1_Unstructure_backward  = Order2_Diff1_Unstructure_backward( x, total_dim=1, diff_dim=1)
    order2_Diff1_Unstructure_upwind = Order2_Diff1_Unstructure_Upwind( x, total_dim=1, diff_dim=1)

    u = torch.sin(2*np.pi*x).unsqueeze(0)
    u_diff1_forward  = order2_Diff1_Unstructure_forward(u)
    print(u_diff1_forward)
    u_diff1_backward = order2_Diff1_Unstructure_backward(u)
    print(u_diff1_backward)
    u_diff1_upwind = order2_Diff1_Unstructure_upwind(u,u)
    
    
    plt.plot(x, u.squeeze())
    # plt.plot(x, u_diff1_forward.squeeze())
    # plt.plot(x, u_diff1_backward.squeeze())
    plt.plot(x, u_diff1_upwind.squeeze())

    
    
    
    


if __name__ == "__main__":
    main()


# %%
