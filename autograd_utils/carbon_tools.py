#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:55:27 2022

@author: yaoyichen
"""

import numpy as np
import torch
import os
from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single,get_uvw
import datetime
import netCDF4 as nc

def construct_state_with_cinit(c_init, flux_all,c_all, u_all, v_all,w_all):
    c_all_temp = torch.cat([c_init,c_all[1::,:,:,:]],dim = 0)
    state_all = torch.stack([flux_all, c_all_temp, u_all, v_all,w_all]).permute([1,0,2,3,4])
    return state_all
    

def construct_state_with_fluxbottom(flux_bottom, flux_all,c_all, u_all, v_all,w_all): 
    f_all_temp = torch.cat([flux_bottom,f_all[:,:,:,1::]],dim = 3)
    state_all = torch.stack([f_all_temp, c_all, u_all, v_all,w_all]).permute([1,0,2,3,4])
    return state_all

def construct_state_with_cinitfluxbottom(c_init,flux_bottom, u_all, v_all,w_all): 
    """
    应该会成为default配置，输入c_init和flux_bottom， 共同构建 state_all
    
    state_all [time, variable, longitude, latitude, height]
    """
    f_all = torch.zeros(u_all.shape)
    c_all = torch.zeros(u_all.shape)  
    
    c_all_temp = torch.cat([c_init,c_all[1::,:,:,:]],dim = 0)
    f_all_temp = torch.cat([flux_bottom,f_all[:,:,:,1::]],dim = 3)
    state_all = torch.stack([f_all_temp, c_all_temp, u_all, v_all,w_all]).permute([1,0,2,3,4])
    return state_all


def construct_state_with_cinitfluxbottom_flux1time(c_init, bottom_flux_one, u_all, v_all,w_all):
    "flux 和 时间无关 "    
    f_all_height = torch.zeros(u_all.shape[0:3])
    bottom_flux = torch.cat([bottom_flux_one, f_all_height[1:,:,:]], dim = 0)
    bottom_flux_unsqueeze = bottom_flux.unsqueeze(-1)
    
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux_unsqueeze, u_all, v_all,w_all)
    return state_all


def construct_state_with_cinitfluxbottom_fluxalltime(c_init, bottom_flux_one, u_all, v_all,w_all):
    "flux 和 时间无关 "    
    bottom_flux = bottom_flux_one.repeat([len(u_all),1,1])
    bottom_flux_unsqueeze = bottom_flux.unsqueeze(-1)
    
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux_unsqueeze, u_all, v_all,w_all)
    return state_all


def construct_c_init():

    """
    读入初始的 浓度数据，可以没有
    
    返回格式 [1, 72, 46, 3])
    """
    folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    file_name = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
    variable_list = ["SpeciesConc_CO2"]
    [c] = get_variable_Merra2_3d_single(folder_name, file_name, latitude_dim = 46, longitude_dim = 72, variable_list = variable_list)
    
    # c_all = torch.tensor(c, dtype= torch.float32).repeat([len(u_all),1,1,1])[:,:,:,0:preserve_layers]
    return c[np.newaxis,:]
    

# def construct_bottom_flux(folder_name):
#     """
#     从处理过的 
    
#     返回格式 [1, 72, 46, 3])
#     """
    #folder_name = "/Users/yaoyichen/dataset/carbon/carbonTracker/CT2019B.flux1x1.201807_reshape.npy"
    # bottom_flux = np.load(folder_name)
    # # c_all = torch.tensor(c, dtype= torch.float32).repeat([len(u_all),1,1,1])[:,:,:,0:preserve_layers]
    # return bottom_flux

    

def get_c_init_xco2(source_config, constant_value):
    """
    初始化二维 xco2
    """
    if(source_config == "init_constant"):
        c_init = constant_value*torch.ones([1, 72, 46, 1])
        c_init = c_init.to(torch.float32)
        return c_init
    

def  get_c_init( source_config, folder_name=None, 
                                file_name = None, 
                                vertical_indexs = None,
                                constant_value = 0.0004,
                                add_mean = None):
    """
    add_mean 是否需要叠加高度方向的分布
    """
    if(source_config == "init_constant"):
        "一開始浓度值为平均值"
        c_init = constant_value*torch.ones([1, 72, 46, len(vertical_indexs)])
        if(add_mean is not None):
            c_init = c_init + add_mean[vertical_indexs]
            c_init = c_init.to(torch.float32)
            
        
    if(source_config == "from_file"):
        "从文件读取"
        c_init = torch.load(os.path.join(folder_name,file_name))
        
        
    if(source_config == "geos-chem"):
        """
        读入初始的 浓度数据，可以没有
        
        返回格式 [1, 72, 46, 3])
        """
        # folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
        # file_name = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
        variable_list = ["SpeciesConc_CO2"]
        [c] = get_variable_Merra2_3d_single(folder_name, file_name, latitude_dim = 46, longitude_dim = 72, variable_list = variable_list)
        
        c_init_all = torch.tensor(c[np.newaxis,:])
        c_init = c_init_all[:,:,:,vertical_indexs]
        
    return c_init


def get_bottom_flux(source_config, time_len = None, file_name = None):
    if(source_config == "init_constant"):
        "一開始浓度值为平均值"
        bottom_flux = 0.0*torch.ones([time_len, 72, 46, 1])
        
    if(source_config == "from_file"):
        "从文件读取"
        bottom_flux = torch.load(file_name)
        
    if(source_config == "carbon_tracker"):
        "读取carbon tracker的文件"
        one_bottom_flux = np.load(file_name)
        bottom_flux = torch.tensor(one_bottom_flux, dtype= torch.float32).repeat([time_len,1,1])
        bottom_flux = bottom_flux.unsqueeze(-1)
        
        
    if(source_config == "carbon_tracker_layer1"):
        "读取carbon tracker的文件"
        one_bottom_flux = np.load(file_name)
        bottom_flux = torch.tensor(one_bottom_flux, dtype= torch.float32).repeat([time_len,1,1])
        bottom_flux = bottom_flux.unsqueeze(-1)
        
        
    if(source_config == "geos-chem_03"):
        """
        guodong 提供的数据，单位为 kg m-2 s-1
        """
        df = nc.Dataset(file_name)
        one_bottom_flux = df.variables["Sflx"][:]
        bottom_flux = torch.tensor(one_bottom_flux, dtype= torch.float32).repeat([time_len,1,1])
        bottom_flux = bottom_flux.unsqueeze(-1)
        # 单位转化
        bottom_flux = bottom_flux/0.044
        bottom_flux = torch.permute(bottom_flux,[0,2,1,3])
        
    return bottom_flux


def construct_Merra2_initial_state_3d(folder_name,file_name, year, month,day, last_day = 1,interval_minutes = 20, preserve_layers = None):
    """
    初始化网格向量，初始化时间轴
    """
    # 读入数据, 每个网格的 map_factor
    # 1度对应的距离
    dx = (6378)*1000 * 2.0*np.pi / 360.0
    dy = dx

    # folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
    # file_name = "GEOSChem.StateMet.20190701_0000z.nc4"
    long_vector, lati_vector = get_variable_Merra2_vector_single(folder_name, file_name,variable_list = ["lon","lat"])

    vector_x = torch.tensor(long_vector * dx, dtype = torch.float32)
    vector_y = torch.tensor(lati_vector * dy, dtype = torch.float32)
    dz = 1
    vector_z = -1.0*torch.tensor([0.9925,0.9775,	0.9625,	0.9475,	0.9325,	0.9175,	0.9025,	0.8875,	0.8725,
    	0.8575,	0.8425,	0.8275,	0.8100,	0.7875,	0.7625,	0.7375,	0.7125,	0.6875,	0.6563,	0.6188,	0.5813,
        	0.5438,	0.5063,	0.4688,	0.4313,	0.3938,	0.3563,	0.3128,	0.2665,	0.2265,	0.1925,	0.1637,
            	0.1391,	0.1183,	0.1005,	0.0854,	0.0675,	0.0483,	0.0343,	0.0241,	0.0145,	0.0067,	0.0029,
                	0.0011,	0.0004,	0.0001,0.0000])[0:preserve_layers]*100000
    
    # print(vector_x, vector_y, vector_z)

    grid_x, grid_y, grid_z = torch.meshgrid(vector_x, vector_y, vector_z)
    map_factor = 1.0 / (torch.cos(2 * np.pi / 360.0 * grid_y / dy))


    latitude_cut_factor = 3.0
    map_factor[map_factor > latitude_cut_factor] =   latitude_cut_factor
    map_factor[map_factor < 0.0] =  latitude_cut_factor

    grid_info = (long_vector,lati_vector,
                 dx, dy, dz, 
                 grid_x, grid_y, grid_z, 
                 vector_x, vector_y, vector_z, map_factor)

    #### time info, 20分钟间隔, 7天  ####
    delta_second = interval_minutes*60
    nt_time = int(60//interval_minutes*24*last_day) - 1
    time_vector = torch.linspace(0., delta_second * nt_time, nt_time + 1)

    time_string = [datetime.datetime(year, month,day) + datetime.timedelta(seconds=time_step*delta_second)
                   for time_step in range(nt_time + 1)]

    time_info = time_vector, time_string, nt_time
    return grid_info, time_info









def construct_uvw_all_3d_zero():
    u_all = torch.zeros([432, 72, 46, 47])
    v_all = torch.zeros([432, 72, 46, 47])
    w_all = torch.zeros([432, 72, 46, 47])
    pbl_top = torch.zeros([432, 72, 46, 1])
    return u_all, v_all, w_all, pbl_top

