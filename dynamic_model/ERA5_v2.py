#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:22:08 2021

@author: yaoyichen
"""

import torch
import torch.nn as nn
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
from dynamic_model.differ_module import Order2_Diff1_Unstructure, Order2_Diff1_Unstructure_Period
from dynamic_model.differ_module import Order2_Diff2_Unstructure, Order2_Diff2_Unstructure_Period
import datetime

from dynamic_model.differ_module import Order2_Diff1_Unstructure_Perioid_Upwind,Order2_Diff1_Unstructure_Upwind


def get_variable(foldername, datestr):
    file_name = "/Users/yaoyichen/dataset/era5/old/myfile19.nc"
    df = nc.Dataset(file_name)
    longitude = df.variables["longitude"][:]
    latitude = df.variables["latitude"][:]

    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    print(f"longitude:{longitude}")
    print(f"latitude:{latitude}")
    u = np.transpose(np.asarray(df.variables["u"][:]), (0, 2, 1))
    v = np.transpose(np.asarray(df.variables["v"][:]), (0, 2, 1))
    phi = np.transpose(np.asarray(df.variables["z"][:]), (0, 2, 1))

    # latitude = np.hstack([latitude[0], latitude[10:-10], latitude[-1]])

    # u = np.concatenate([np.expand_dims(u[:, :, 0], axis=2),
    #                     u[:, :, 10:-10], np.expand_dims(u[:, :, -1], axis=2)], axis=2)
    # v = np.concatenate([np.expand_dims(v[:, :, 0], axis=2),
    #                     v[:, :, 10:-10], np.expand_dims(v[:, :, -1], axis=2)], axis=2)
    # phi = np.concatenate([np.expand_dims(phi[:, :, 0], axis=2),
    #                       phi[:, :, 10:-10], np.expand_dims(phi[:, :, -1], axis=2)], axis=2)

    # latitude = latitude[10:-10]

    # u = u[:, :, 10:-10]
    # v = v[:, :, 10:-10]
    # phi = phi[:, :, 10:-10]

    return longitude, latitude, u, v, phi


class ERA5_pressure(nn.Module):
    def __init__(self, grid_info):

        super(ERA5_pressure, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y, self.map_factor = grid_info

        self.map_factor = torch.tensor(self.map_factor).to(torch.float32)

        print(f"map_factor:{torch.mean(self.map_factor, dim = (0))}")
        self.diff1_x = Order2_Diff1_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff1_y = Order2_Diff1_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

        self.diff2_x = Order2_Diff2_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        self.diff2_y = Order2_Diff2_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)

        self.m2_diff_1_m = (self.map_factor**2 *
                            self.diff1_y(1.0/self.map_factor.unsqueeze(0))).squeeze().to(torch.float32)

        self.f = torch.tensor(
            2*7.0e-5*np.sin(2*np.pi/360.0*self.grid_y/self.dy)).to(torch.float32)

    def forward(self, t, state):
        u, v, phi = state
        u, v, phi = u.unsqueeze(0), v.unsqueeze(0), phi.unsqueeze(0)

        f_star = self.f - u * self.m2_diff_1_m

        du = -1.0 * self.map_factor * u * self.diff1_x(u) - v * self.diff1_y(u) - \
            self.map_factor * self.diff1_x(phi) + 1e-3*(self.map_factor**2*self.diff2_x(u) +
                                                        self.diff2_y(u)) + f_star * v

        dv = -1.0*self.map_factor * u * self.diff1_x(v) - v * self.diff1_y(v) - \
            self.diff1_y(phi) + 1e-3 * (self.map_factor**2*self.diff2_x(v) +
                                        self.diff2_y(v)) - f_star * u

        # dphi = - u*self.diff1_x(phi) - v*self.diff1_y(phi)
        # dphi = -self.map_factor * u*self.diff1_x(phi) - v*self.diff1_y(phi)
        # - phi * (self.map_factor * self.diff1_x(u) + self.diff1_y(v))

        dphi = -self.map_factor * self.diff1_x(phi * u) - self.diff1_y(phi * v) \
            + 1e-3 * (self.map_factor**2*self.diff2_x(phi) +
                      self.diff2_y(phi))

        # dphi = -u*self.map_factor * self.diff1_x(phi) - v*self.diff1_y(phi)

        result = torch.stack([du.squeeze(), dv.squeeze(), dphi.squeeze()])

        result[0:2, :, 0] = 0.0
        result[0:2, :, -1] = 0.0

        return result
    
    
def construct_ERA5_v2_initial_state_1day():
    x, y, u, v, phi = get_variable("a", "b")
    return u[4,:,:], v[4,:,:], phi[4,:,:]
    


def construct_ERA5_v2_initial_state():
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离

    dx = (6371 + 5)*1000 * 2.0*np.pi / 360.0
    dy = dx
    x, y, u, v, phi = get_variable("a", "b")
    # print(u.shape, v.shape, phi.shape)

    print(f"x:{x} ,y:{y}")
    vector_x = torch.tensor(x * dx)
    vector_y = torch.tensor(y * dy)
    grid_x, grid_y = torch.meshgrid(vector_x, vector_y)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))

    map_factor[map_factor > 3.0] = 3.0
    map_factor[map_factor < 3.0] = 3.0

    # map_factor[:] = 1.0

    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor)

    state0_true = np.vstack(
        [np.expand_dims(u[0, :, :], axis=0),
         np.expand_dims(v[0, :, :], axis=0),
         np.expand_dims(phi[0, :, :], axis=0)])
    state0_true = torch.tensor(state0_true, dtype=torch.float32)

    state = np.vstack(
        [np.expand_dims(u[-1, :, :], axis=0),
         np.expand_dims(v[-1, :, :], axis=0),
         np.expand_dims(phi[-1, :, :], axis=0)])
    state = torch.tensor(state, dtype=torch.float32)

    state_info = (state, state0_true)

    #### time info  ####
    delta_second = 60*10
    nt_time = int(6*24*2)
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1)

    time_string = [datetime.datetime(2020, 9, 23) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]

    ind_obs = torch.arange(6*6, 6*24*2+1, 6*6)
    nt_obs = len(ind_obs)

    time_info = time_vector, time_string, nt_time, ind_obs, nt_obs
    return grid_info, state_info, time_info




class SP_Filter(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, half_padding=1, dtype = "float32"):
        super(SP_Filter, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv2d(
            1, 1, (self.kernel_size, self.kernel_size), padding=(1, 1), padding_mode="circular")

        if(dtype == "float32" ):
            weights = torch.tensor(
                [[1.0/16.0,  2.0/16.0,  1.0/16.0], [2.0/16.0,  4.0/16.0,  2.0/16.0], [1.0/16.0,  2.0/16.0,  1.0/16.0]], dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
            bias = torch.tensor([0.0], dtype=torch.float32)
        if(dtype == "float64" ):
            weights = torch.tensor(
                [[1.0/16.0,  2.0/16.0,  1.0/16.0], [2.0/16.0,  4.0/16.0,  2.0/16.0], [1.0/16.0,  2.0/16.0,  1.0/16.0]], dtype=torch.float64).view(1, 1, self.kernel_size, self.kernel_size)
            bias = torch.tensor([0.0], dtype=torch.float64)
            
            
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



class Weatherbench_QG(nn.Module):
    def __init__(self, grid_info):

        super(Weatherbench_QG, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y, self.map_factor = grid_info

        self.map_factor = torch.tensor(self.map_factor).to(torch.float64)

        print(f"map_factor:{torch.mean(self.map_factor, dim = (0))}")
        
        self.diff1_x_upwind =  Order2_Diff1_Unstructure_Perioid_Upwind(
                self.vector_x, total_dim=2, diff_dim=1, dtype = "float64")
        
        self.diff1_y_upwind =   Order2_Diff1_Unstructure_Upwind(
                self.vector_y, total_dim=2, diff_dim=2, dtype = "float64")
        
        
        self.diff1_x = Order2_Diff1_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1, dtype = "float64")
        
        self.diff1_y = Order2_Diff1_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2, dtype = "float64")

        # self.diff2_x = Order2_Diff2_Unstructure_Period(
        #     self.vector_x, total_dim=2, diff_dim=1, dtype = "float64")
        
        self.diff2_y = Order2_Diff2_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)
    
        self.m2_diff_1_m = (self.map_factor**2 *
                            self.diff1_y(1.0/self.map_factor.unsqueeze(0))).squeeze().to(torch.float64)

        self.f = torch.tensor(
            2*7.0e-5*np.sin(2*np.pi/360.0*self.grid_y/self.dy)).to(torch.float64)

    def forward(self, t, state):
        u, v, phi = state
        u, v, phi = u.unsqueeze(0), v.unsqueeze(0), phi.unsqueeze(0)
        
        # u = torch.clip(u,min = -30, max = 30)
        # v = torch.clip(v,min = -20, max = 20)

        # f_star = self.f   - u * self.m2_diff_1_m
        f_star = self.f 

        du = -1.0 * self.map_factor * u * self.diff1_x(u)  - v * self.diff1_y(u) 
        # + f_star * v
         # - \
        #     self.map_factor * self.diff1_x(phi) + f_star * v
            
            # + 1e-2*(self.map_factor**2*self.diff2_x(u) + \
            #                                             self.diff2_y(u)) 

        dv = -1.0*self.map_factor * u * self.diff1_x(v) - v * self.diff1_y(v) 
        # - f_star * u
            # self.diff1_y(phi) 
            
            # + 1e-2 * (self.map_factor**2*self.diff2_x(v) + \
            #                             self.diff2_y(v)) 

        # dphi = - u*self.diff1_x(phi) - v*self.diff1_y(phi)
        # dphi = -self.map_factor * u*self.diff1_x(phi) - v*self.diff1_y(phi)
        # - phi * (self.map_factor * self.diff1_x(u) + self.diff1_y(v))

        # dphi = -self.map_factor * self.diff1_x_upwind( u*phi,u) - self.diff1_y( v*phi) \
        # - self.map_factor * phi* self.diff1_x_upwind( u,u) - phi* self.diff1_y( v)
        
        # dphi = -self.map_factor * self.diff1_x(u*phi) - self.diff1_y( v*phi) \
        # - self.map_factor * phi* self.diff1_x( u) - phi* self.diff1_y( v)
        # print(f"phi:{phi.dtype}, u:{u.dtype},v:{v.dtype}")
        dphi = - self.map_factor * u* self.diff1_x_upwind(phi,u) - v* self.diff1_y_upwind( phi,v) 
        
        # du[:] = 0.0
        # dv[:] = 0.0
        # dphi[:] = 0.0
            
        
        # print(f"du:{du.dtype}, dv:{dv.dtype},dphi:{dphi.dtype}")
        
        
        # dphi = 0.2*dphi
        # dphi = -self.map_factor * self.diff1_x_upwind(u* phi,u) - self.diff1_y_upwind(v* phi,v)
        # \
        #     

        # dphi = -u*self.map_factor * self.diff1_x(phi) - v*self.diff1_y(phi)

        result = torch.stack([du.squeeze(), dv.squeeze(), dphi.squeeze()])
        result = result.to(torch.float64)
        result[0:3, :, 0] = 0.0
        result[0:3, :, -1] = 0.0

        return result


class Weatherbench_QG_temp(nn.Module):
    def __init__(self, grid_info):
        super(Weatherbench_QG_temp, self).__init__()
        self.dx, self.dy, self.grid_x, self.grid_y, self.vector_x, self.vector_y, self.map_factor = grid_info

        self.map_factor = torch.tensor(self.map_factor).to(torch.float32)
        
        # print(f"map_factor:{torch.mean(self.map_factor, dim = (0))}")
        self.diff1_x_upwind =  Order2_Diff1_Unstructure_Perioid_Upwind(
                self.vector_x, total_dim=2, diff_dim=1)

        self.diff1_y_upwind =   Order2_Diff1_Unstructure_Upwind(
                self.vector_y, total_dim=2, diff_dim=2)
        
        
        self.diff1_x = Order2_Diff1_Unstructure_Period(
            self.vector_x, total_dim=2, diff_dim=1)
        
        self.diff1_y = Order2_Diff1_Unstructure(
            self.vector_y, total_dim=2, diff_dim=2)


        self.m2_diff_1_m = (self.map_factor**2 *
                            self.diff1_y(1.0/self.map_factor.unsqueeze(0))).squeeze()

        self.f = torch.tensor(
            2*7.0e-5*np.sin(2*np.pi/360.0*self.grid_y/self.dy)).to(torch.float32)

    def forward(self, t, state):
        u, v, phi = state
        u, v, phi = u.unsqueeze(0), v.unsqueeze(0), phi.unsqueeze(0)

        f_star = self.f - u * self.m2_diff_1_m

        du = -1.0 * self.map_factor * u * self.diff1_x_upwind(u,u) - v * self.diff1_y_upwind(u,v) + f_star * v
        # - \
        #     self.map_factor * self.diff1_x_upwind(phi, u)  

        dv = -1.0*self.map_factor * u * self.diff1_x_upwind(v,u) - v * self.diff1_y_upwind(v,v) - f_star * u
        # - \
        #     self.diff1_y_upwind(phi,v)  

        dphi = -self.map_factor * self.diff1_x_upwind(phi * u, u) - self.diff1_y_upwind(phi * v, v) 
        
        # print(f"du:{du.dtype}, dv:{dv.dtype},dphi:{dphi.dtype},")
        # dphi = -u*self.map_factor * self.diff1_x(phi) - v*self.diff1_y(phi)

        result = torch.stack([du.squeeze(), dv.squeeze(), dphi.squeeze()])

        result[0:2, :, 0] = 0.0
        result[0:2, :, -1] = 0.0

        return result
    
    


def construct_weatherbench_initial_state():
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离

    dx = (6371 + 5)*1000 * 2.0*np.pi / 360.0
    dy = dx
    x = np.asarray([  0.   ,   5.625,  11.25 ,  16.875,  22.5  ,  28.125,  33.75 ,
            39.375,  45.   ,  50.625,  56.25 ,  61.875,  67.5  ,  73.125,
            78.75 ,  84.375,  90.   ,  95.625, 101.25 , 106.875, 112.5  ,
           118.125, 123.75 , 129.375, 135.   , 140.625, 146.25 , 151.875,
           157.5  , 163.125, 168.75 , 174.375, 180.   , 185.625, 191.25 ,
           196.875, 202.5  , 208.125, 213.75 , 219.375, 225.   , 230.625,
           236.25 , 241.875, 247.5  , 253.125, 258.75 , 264.375, 270.   ,
           275.625, 281.25 , 286.875, 292.5  , 298.125, 303.75 , 309.375,
           315.   , 320.625, 326.25 , 331.875, 337.5  , 343.125, 348.75 ,
           354.375])
    
    y = np.asarray([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625,
           -53.4375, -47.8125, -42.1875, -36.5625, -30.9375, -25.3125,
           -19.6875, -14.0625,  -8.4375,  -2.8125,   2.8125,   8.4375,
            14.0625,  19.6875,  25.3125,  30.9375,  36.5625,  42.1875,
            47.8125,  53.4375,  59.0625,  64.6875,  70.3125,  75.9375,
            81.5625,  87.1875])

    print(f"x:{x} ,y:{y}")
    vector_x = torch.tensor(x * dx)
    vector_y = torch.tensor(y * dy)
    grid_x, grid_y = torch.meshgrid(vector_x, vector_y)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))

    map_factor[map_factor > 2.0] = 2.0
    map_factor[map_factor < 0.0] = 2.0

    # map_factor[:] = 1.0

    grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y, map_factor)

    # state0_true = np.vstack(
    #     [np.expand_dims(u[0, :, :], axis=0),
    #      np.expand_dims(v[0, :, :], axis=0),
    #      np.expand_dims(phi[0, :, :], axis=0)])
    # state0_true = torch.tensor(state0_true, dtype=torch.float32)

    # state = np.vstack(
    #     [np.expand_dims(u[-1, :, :], axis=0),
    #      np.expand_dims(v[-1, :, :], axis=0),
    #      np.expand_dims(phi[-1, :, :], axis=0)])
    # state = torch.tensor(state, dtype=torch.float32)

    # state_info = (state, state0_true)

    #### time info  ####
    delta_second = 60*60
    nt_time = int(1*24*1)
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1).to(torch.float64)

    time_string = [datetime.datetime(2020, 9, 23) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]


    time_info = time_vector, time_string, nt_time
    return grid_info, time_info




def write_netcdf(data_, ref_nc_name, output_nc_name, time_string, plot_interval):

    df = nc.Dataset(ref_nc_name)

    original_nx, original_ny, original_nt = df.dimensions[
        "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
    new_nx, new_ny = original_nx, original_ny

    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', new_nx)

    ncout.createDimension('latitude', new_ny)
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))

    lonvar.setncattr('units', df.variables["longitude"].units)
    latvar.setncattr('units', df.variables["latitude"].units)
    timevar.setncattr('units', df.variables["time"].units)

    total_len = len(time_string)

    lonvar[:] = df.variables["longitude"][:]

    latvar[:] = df.variables["latitude"][:]

    calendar = 'standard'
    timevar[:] = nc.date2num(time_string[
        0:total_len:plot_interval], units=df.variables["time"].units, calendar=calendar)

    u = ncout.createVariable("u", 'float32',
                             ('time', 'latitude', 'longitude'))
    v = ncout.createVariable("v", 'float32',
                             ('time', 'latitude', 'longitude'))
    z = ncout.createVariable("z", 'float32',
                             ('time', 'latitude', 'longitude'))

    u[:] = np.transpose(data_[0:total_len:plot_interval, 0, :, :], (0, 2, 1))
    v[:] = np.transpose(data_[0:total_len:plot_interval, 1, :, :], (0, 2, 1))
    z[:] = np.transpose(data_[0:total_len:plot_interval, 2, :, :], (0, 2, 1))

    ncout.close()
    return None


def filter_longitude(state_, nx, ny, percentage=10,dtype = "float32"):
    # a, b, c = state_
    nx_h = nx // 2
    state_filter = torch.empty(state_.shape)

    for index, variable in enumerate(state_):
        b = torch.fft.fft2(variable)
        b[nx_h - nx//percentage: nx_h + nx//percentage, :] = 0.0
        if(dtype == "float32"):
            c = torch.fft.ifft2(b).float()
        if(dtype == "float64"):
                c = torch.fft.ifft2(b).to(torch.float64)
        state_filter[index, :, :] = c

    return state_filter


def filter_latitude(state_, nx, ny, percentage=10,dtype = "float32"):
    nx_h = nx//2
    # a, b, c = state_
    state_filter = torch.empty(state_.shape)

    for index, variable in enumerate(state_):
        b = torch.flip(variable[nx_h::, :], dims=(1,))
        c = torch.cat([variable[0:nx_h, ::], b[:, 1:-1]], dim=1)
        d = torch.fft.fft2(c)
        d[:, ny - 2*ny//percentage: ny + 2*ny//percentage] = 0.0
        
        
        if(dtype == "float32"):
            e = torch.fft.ifft2(d).float()
        if(dtype == "float64"):
            e = torch.fft.ifft2(d).to(torch.float64)
        
        f1 = e[0:nx_h, 0:ny + 1]
        f2 = torch.empty(f1.shape)
        f2[:, 0] = f1[:, 0]
        f2[:, -1] = f1[:, -1]
        f2[:, 1:-1] = torch.flip(e[0:nx_h, ny + 1: 2*ny], dims=(1,))

        g = torch.cat([f1, f2], dim=0)
        state_filter[index, :, :] = g
    return state_filter
