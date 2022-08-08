#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 19:02:53 2022

@author: yaoyichen
"""
#%%
import sys
sys.path.append("..") 
import os
import torch
import torch.nn as nn
import time
import numpy as np
from tools.file_helper import FileUtils
from dynamic_model.Transport import CTM_Model_3D
from dynamic_model.Transport import ctm_simulation_3d_mixing,SP_Filter_3D
from autograd_utils.file_output import write_carbon_netcdf_3d_geoschem, write_carbon_netcdf_2d_geoschem
from autograd_utils.carbon_tools import construct_state_with_cinitfluxbottom
from autograd_utils.carbon_tools  import get_bottom_flux, get_c_init, get_c_init_xco2
from autograd_utils.carbon_tools  import construct_Merra2_initial_state_3d 
from autograd_utils.geoschem_tools import construct_uvw_all_3d,generate_vertical_info,regenerate_vertcal_state
from autograd_utils.geoschem_tools  import calculate_x_result
from data_prepare.generate_observation_data import generate_obspack_data,generate_satellite_data
import argparse

from autograd_utils.statistical_tools import  calcuate_mask_score


class Args:
    year = 2019
    month = 7
    day = 1
    last_day = 5       # 仿真天数
    interval_minutes = 30    # 仿真时间间隔，需要与merra2文件夹下的结果保持一致。
    device = "cpu"
    
    if_mixing = False  # 是否开启边界层混合 
    sim_dimension = 3  #  2,3: 二维还是3维
    
    layer_type = "layer_1" # 仿真的层数配置 "layer_9","layer_47","layer_1"
    if_plot_result = True     # 是否需要保存仿真结果
    plot_interval = 24        # 每多少个时间步保存一份结果, 30min的仿真，对应的是1天

    
    # data_folder= '/Users/yaoyichen/dataset/auto_experiment/experiment_0/'
    # result_folder = '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/result_cite_inversion/'
    
    # data_folder= '/home/eason.yyc/data/auto_experiment/experiment_0/'
    # result_folder =  '/home/eason.yyc/data/auto_experiment/experiment_0/result_3B/'
    
    geoschem_co2_file = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
    
    problem_type = "inversion"   # "forward_simulaiton", "assimilation" ,"inversion"
    
    def print_attributes(self):
        result_mapping = {}
        result_mapping["year"] = self.year
        result_mapping["month"] = self.month
        result_mapping["day"] = self.day
        result_mapping["interval_minutes"] = self.interval_minutes
        result_mapping["if_mixing"] = self.if_mixing
        result_mapping["sim_dimension"] = self.sim_dimension
        result_mapping["layer_type"] = self.layer_type
        result_mapping["problem_type"] = self.problem_type
        
        print(result_mapping)
        
        return result_mapping
                
args = Args()
print(args.print_attributes())


def parse_args():
    parser = argparse.ArgumentParser(
        description='CTM')
    parser.add_argument('--year', type=int, default = 2019 )
    parser.add_argument('--month', type=int, default = 7)
    parser.add_argument('--day', type=int, default = 1)
    parser.add_argument('--last_day', type=int, default = 5)
    parser.add_argument('--interval_minutes', type=int, default = 30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--if_mixing_str', choices=('True', 'False'))
    parser.add_argument('--sim_dimension', type=int, default = 2)
    parser.add_argument('--layer_type', type=str, choices=('layer_47', 'layer_9','layer_1'))
    parser.add_argument('--if_plot_result_str', type=str, choices=('True', 'False'))
    parser.add_argument('--plot_interval',  type=int, default = 100 )
    parser.add_argument('--data_folder', type=str)

    parser.add_argument('--sub_data_folder', type=str)
    parser.add_argument('--result_folder', type=str)
    parser.add_argument('--geoschem_co2_file', type=str)
    parser.add_argument('--problem_type', type=str, choices=('forward_simulation', 'inversion','assimilation'))

    parser.add_argument('--experiment_type', type=str, default = "none", choices=('twin', 'real','none'))
    parser.add_argument('--obs_type', type=str, default = "none", choices=('full', 'satellite',"obspack",'none'))

    parser.add_argument('--flux_type', type=str, default = "init_constant", choices=("geos-chem_03","carbon_tracker",'init_constant'))
    parser.add_argument('--flux_file_name', type = str)
    parser.add_argument('--init_flux_file_name', type = str) # 反演时候初始化的通量来源
    parser.add_argument('--cinit_type', type=str, default = "geos-chem", choices=('geos-chem', "init_constant","from_file"))
    parser.add_argument('--cinit_file', type=str, default = None)

    # 以下两个参数, 仅提供给孪生实验使用
    # parser.add_argument('--flux_type_twin', type=str, default = "none", choices=('full', 'satellite',"obspack",'none'))
    # parser.add_argument('--cinit_type_assimilation', type=str, default = "none", choices=('full', 'satellite',"obspack",'none'))
    parser.add_argument('--flux_type_inversion', type=str, default = "init_constant", choices=("geos-chem_03","carbon_tracker",'init_constant',"from_file"))
    parser.add_argument('--cinit_type_assimilation', type=str, default = "geos-chem", choices=('geos-chem', "init_constant","constant_obs"))


    parser.add_argument('--lr_flux', type=float, default = 1e-3)
    parser.add_argument('--lr_cinit', type=float, default = 3e-5)

    parser.add_argument('--iteration_number', type=int, default = 3)
    parser.add_argument('--early_stop_value', type=float, default = 0.5e-6)
    parser.add_argument('--plot_per_iteration', type=int, default = 1)


    parser.add_argument('--if_background', type=str, default = "False", choices=("False",'True'))
    parser.add_argument('--background_weight', type=float)

    args = parser.parse_args()
    return args
    
args = parse_args()

if(args.if_mixing_str == "True"):
    args.if_mixing = True
if(args.if_mixing_str == "False"):
    args.if_mixing = False
if(args.if_plot_result_str == "True"):
    args.if_plot_result = True
if(args.if_plot_result_str == "False"):
    args.if_plot_result = False

print(args)
print(f"args.if_mixing_str:{args.if_mixing_str}")


# random variable
random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.set_printoptions(precision=5)


simulation_len = int(args.last_day*1440//args.interval_minutes)
merra2_folder = os.path.join(args.data_folder, 'merra2', args.sub_data_folder)
geoschem_folder = os.path.join(args.data_folder,  'geoschem')
satellite_folder = os.path.join(args.data_folder,  'satellite')
obspack_folder = os.path.join(args.data_folder,  'obspack')

FileUtils.makedir(args.result_folder)

#%%
# step1.1 读入气象数据
print("loading merra2 data...")
startswith_string = "GEOSChem.StateMet."
u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine, merra_file_name_list \
    = construct_uvw_all_3d( merra2_folder, startswith_string,
                           args)

#%%
# step1.2 初始化网格信息，以及时间信息
grid_info, time_info = construct_Merra2_initial_state_3d(folder_name = merra2_folder,
                                                         file_name = merra_file_name_list[0],
                                                         year = args.year,
                                                         month = args.month,
                                                         day = args.day,
                                                         last_day = args.last_day,
                                                         interval_minutes = args.interval_minutes)

time_vector, time_string, nt_time = time_info
(longitude_vector,latitude_vector,
             dx, dy, dz, _1, _2, _3, 
             vector_x, vector_y, vector_z_origine, map_factor_origine) = grid_info

# #%%
# # step 1.3 配置高度方向信息，生成特定高度上的状态


vertical_mapping = generate_vertical_info(args.layer_type)

[vector_z, map_factor, u_all, v_all, w_all], weight_z, pbl_top \
    = regenerate_vertcal_state(vector_z_origine, map_factor_origine, 
                              u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine, vertical_mapping)

# # step 1.4 初始化CO2浓度信息， 初始化bottom-up flux 
if(args.cinit_type == "constant"):
    c_init = get_c_init(source_config = "init_constant", 
                        vertical_indexs= vertical_mapping["vertical_indexs"],
                        constant_value = 4e-4)


if(args.cinit_type == "geos-chem"):
    c_init = get_c_init(source_config = "geos-chem", folder_name=geoschem_folder, 
                    file_name = args.geoschem_co2_file, 
                    vertical_indexs = vertical_mapping["vertical_indexs"]  )


if(args.cinit_type == "from_file"): 
    c_init = get_c_init(source_config = "from_file",
                        folder_name= "",
                       # file_name = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_assimilation/assimilation_result_ocoobs_3e-5_cinit019.pt"
                       #file_name = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/real_layer1_assimilation_satellite_20180701/assimilation_result_ocoobs_3e-5_cinit049.pt"
                       file_name = args.cinit_file
                       )


if(args.flux_type == "geos-chem_03"):
    bottom_flux = get_bottom_flux(source_config = "geos-chem_03", 
                                time_len = u_all.shape[0],
                                file_name = os.path.join(args.data_folder, "comparison/"
                                                            "input.nc4"))
if(args.flux_type == "carbon_tracker"):
    bottom_flux = get_bottom_flux(source_config = "carbon_tracker", 
                                time_len = u_all.shape[0],
                                file_name = os.path.join(args.data_folder, 
                                                            "carbontracker/",
                                                            args.flux_file_name))
if(args.flux_type == "init_constant"):
    bottom_flux = get_bottom_flux(source_config = "init_constant", 
                                time_len = u_all.shape[0])

# #%% 
# ## step 1.5 将 c_init, bottom_flux, u_all, v_all,w_all 配置成大向量 state_all
# #生成 state_all: [time, variable, longitude, latitude, height]
# # variable 排列 flux, con, u, v, w
# # state_all.shape: torch.Size([240, 5, 72, 46, 9])
# # c_init.shape: torch.Size([1, 72, 46, 9])
# # bottom_flux.shape: torch.Size([240, 72, 46, 1])


# #%% 开始数值仿真
# # step 2.1 构建CTM 模型
model = CTM_Model_3D(grid_info=( vector_x, vector_y, vector_z, map_factor), 
    dim = args.sim_dimension)
model = model.to(args.device)

# # step 2.2 开始数值仿真
if( args.problem_type == "forward_simulation" ):
    print(c_init.shape,bottom_flux.shape, u_all.shape, v_all.shape, w_all.shape )
    state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

    variable_list = [state_all,time_vector, vector_x, vector_y, vector_z, map_factor]
    for variable in variable_list:
        variable = variable.to(args.device)
    
    with torch.no_grad():
        print("start simulation")
        start_time = time.time()
        total_result = ctm_simulation_3d_mixing(model, time_vector, state_all,pbl_top, 
                                                            vertical_mapping, weight_z,
                                                            if_mixing = args.if_mixing)
        print(f"end simulation, elapse time = {(time.time() - start_time):.2f} s")
    print(total_result.shape)

    
    # #%% step 3.1 写文件
    if(args.if_plot_result):
        write_carbon_netcdf_3d_geoschem(data_= total_result.detach().numpy(),
                      output_nc_name= os.path.join(args.result_folder, f"forward_simulation_3d.nc"),
                      time_string=time_string, plot_interval= args.plot_interval,
                      longitude_vector = longitude_vector,
                      latitude_vector = latitude_vector,
                      vector_z = vector_z)
        
        # 计算高度方向加权后的X结果, 输出2维nc文件
        # x_result = calculate_x_result(total_result, weight_z)

        x_result = total_result.mean(axis = -1).unsqueeze(-1)
        write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                      output_nc_name= os.path.join(args.result_folder, f"forward_simulation_2d.nc"),
                      time_string=time_string, plot_interval= args.plot_interval,
                      longitude_vector = longitude_vector,
                      latitude_vector = latitude_vector)


#%%

"""
以上代码不做修改！！！
"""
#%%

# 读入
def constrcut_obs(obs_type, experiment_type,
                  total_result, weight_z, args):
    
    index_vector = None
    if( obs_type == "satellite"):
        """
        # satellite_filename = os.path.join(args.data_folder,"satellite", file_name)
        # index_vector, satellite_value_vector_np , satellite_time_vector = get_oco(satellite_filename)  
        # satellite_value_vector = torch.tensor(satellite_value_vector_np)/10000/100.
        """

        [index_vector, satellite_value_vector, obs_time_vector, obs_longitude_vector, obs_latitude_vector] = \
            generate_satellite_data(args.year, args.month, args.day, 
                                  args.last_day, args.interval_minutes*60,
                        satellite_folder, args.result_folder, f"satellite_{args.year}{str(args.month).zfill(2)}{str(args.day).zfill(2)}_{str(args.last_day).zfill(2)}")
        
    
    if(obs_type == "obspack"):
        """
        # obspack_filename = os.path.join(args.data_folder,"obspack", file_name)
        # print(obspack_filename)
        # index_vector, obspack_value_vector_np , obspack_time_vector = get_oco(obspack_filename)  
        # obspack_value_vector = torch.tensor(obspack_value_vector_np)
        """
        
        [index_vector, obspack_value_vector, obs_time_vector, obs_longitude_vector, obs_latitude_vector] = \
            generate_obspack_data(args.year, args.month, args.day, 
                                  args.last_day, args.interval_minutes*60,
                        obspack_folder, args.result_folder,   f"obspack_{args.year}{str(args.month).zfill(2)}{str(args.day).zfill(2)}_{str(args.last_day).zfill(2)}")
        
    
    if(experiment_type == "twin"):
        true_obs_tensor_full = calculate_x_result(total_result[:, 1:2, :,:,:], weight_z)
        if(obs_type == "full"):
            true_obs_tensor = true_obs_tensor_full
        if(obs_type == "satellite"):
            true_obs_tensor = true_obs_tensor_full.flatten()[index_vector].detach()
        if(obs_type == "obspack"):
            true_obs_tensor = true_obs_tensor_full.flatten()[index_vector].detach()
            
    
    if(experiment_type == "real"):
        if(obs_type == "satellite"):
            true_obs_tensor = satellite_value_vector
        if(obs_type == "obspack"):
            true_obs_tensor = obspack_value_vector
            
    return true_obs_tensor, index_vector
    

print("here")
"""
同化和反演的公共部分
"""
from torch.optim import Adam
if( (args.problem_type == "assimilation") or (args.problem_type == "inversion" )):
    
    iteration_number = args.iteration_number
    early_stop_value = args.early_stop_value
    # from data_prepare.read_era5_uvwfc import get_oco
    
    # 除了 full, real 的配置，其他都可以。 如果是1层的数据，lr需要相应减小
    obs_type = args.obs_type  # "full", "satellite", "obspack"
    experiment_type = args.experiment_type  #, "twin" , "real"
    
    # 这两份数据，需要能够自动生成
    """
    # satellite_file = "satellite_data_20190701_240_72_46.npz"
    # obspack_file =   "obspack_data_20190701_240_72_46.npz"
    # file_name = None
    # if(obs_type == "satellite"):
    #     file_name = satellite_file
    # if(obs_type == "obspack"):
    #     file_name = obspack_file
    """
    if(experiment_type == "real"):
        total_result = None
    
        
    if(experiment_type == "twin"):
        print("start simulation")
        state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux, u_all, v_all,w_all)

        variable_list = [state_all,time_vector, vector_x, vector_y, vector_z, map_factor]
        for variable in variable_list:
            variable = variable.to(args.device)
        
        with torch.no_grad():
            start_time = time.time()
            total_result = ctm_simulation_3d_mixing(model, time_vector, state_all,pbl_top, 
                                                                vertical_mapping, weight_z,
                                                                if_mixing = args.if_mixing)
        print("end twin simulation")
    
    
    true_obs_tensor, index_vector =  constrcut_obs(obs_type, 
                                                    experiment_type,
                                                    total_result,
                                                    weight_z, args)
    

        
    
    
    
    

    

if( args.problem_type == "assimilation" ):
    
    # 读入均值方差数据，本来想给多层的的同化提供些先验的，暂时hold
    with open(os.path.join(args.data_folder,"carbontracker/",'ct_mean_std_profile.npy'), 'rb') as f:
        merra2_mean_vector = np.load(f)
        merra2_std_vector  = np.load(f)
        merra2_mean_vector_torch = torch.tensor(merra2_mean_vector[vertical_mapping["vertical_indexs"]]).to(torch.float32)
        merra2_std_vector_torch  = torch.tensor(merra2_std_vector[vertical_mapping["vertical_indexs"]]).to(torch.float32)
    
    """
    """
    if(args.cinit_type_assimilation == "constant_obs"):
        c_init_pred = get_c_init(source_config = "init_constant", 
                            vertical_indexs= vertical_mapping["vertical_indexs"],
                            constant_value = torch.mean(true_obs_tensor))
    
    c_init_pred.requires_grad = True
    
    
    init_lr = args.lr_cinit
    
    optimizer_cinit = Adam([c_init_pred], lr=init_lr)
    print(f"init_lr:{init_lr}")
    
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    
    filter_op = SP_Filter_3D()
    
    for iteration in range(iteration_number):
        # 需要重新赋值
        print(f"iteration:{iteration}")
        start_time = time.time()
        
        state_all = construct_state_with_cinitfluxbottom(c_init_pred, bottom_flux, u_all, v_all, w_all)
        
        state_full_pred = ctm_simulation_3d_mixing(model, time_vector, state_all,pbl_top, 
                                                           vertical_mapping,weight_z,
                                                           if_mixing=args.if_mixing)
        
        # 变量选择浓度, 在变量和高度方向平均
        pred_obs_tensor_full = calculate_x_result(state_full_pred[:, 1:2, :,:,:], weight_z)
        if(obs_type == "full"):
            pred_obs_tensor = pred_obs_tensor_full
        if(obs_type in ["obspack","satellite"]):
            pred_obs_tensor = pred_obs_tensor_full.flatten()[index_vector]
            
            
        loss_mse = criterion_mse(pred_obs_tensor, true_obs_tensor)
        loss_mae = criterion_mae(pred_obs_tensor, true_obs_tensor)
        print( f"iteration:{iteration}, conct mse:{loss_mse.item():.3g}, mae:{loss_mae.item():.3g}")
        if(loss_mae <= early_stop_value):
            print(f"early stop at iteration:{iteration}, mae:{loss_mae}")
            break
        
        # if(experiment_type == "twin"):
        #     cinit_mae = criterion_mae(c_init_pred, c_init_true)
        #     cinit_mse = criterion_mse(c_init_pred, c_init_true)
        #     print( f"iteration:{iteration}, cinit mse:{cinit_mse.item():.3g}, mae:{cinit_mae.item():.3g}")
        
        
        # c_init_pred_mean = c_init_pred.mean(dim = (0,1,2))
        # c_init_pred_std  = c_init_pred.std(dim = (0,1,2))
        # c_init_pred_mean_res = c_init_pred_mean - torch.mean(c_init_pred_mean)
        
        # loss_mean_profile = torch.sqrt(criterion_mse(c_init_pred_mean_res, merra2_mean_vector_torch))
        # loss_std_profile  = torch.sqrt(criterion_mse(c_init_pred_std, merra2_std_vector_torch))
        # print(f"loss_mean_profile:{loss_mean_profile:.3g}, loss_std_profile:{loss_std_profile:.3g}")
        
        if(args.if_background == "True" and args.background_weight):
            background_loss = args.background_weight*criterion_mse(c_init_pred, filter_op(c_init_pred) )
            print( f"iteration:{iteration}, background_loss:{background_loss.item():.3g}")
        
        if(args.if_background == "True"):
            total_loss = loss_mse + background_loss
        else:
            total_loss = loss_mse
        
        optimizer_cinit.zero_grad()
        total_loss.backward()
        optimizer_cinit.step()
        print(f"elapse time:{time.time() - start_time: .3e}")
        
        if(args.if_plot_result and  ((iteration+1)%args.plot_per_iteration ==0)):
            write_carbon_netcdf_3d_geoschem(data_= state_full_pred.detach().numpy(),
                         output_nc_name= os.path.join(args.result_folder, 
                         f"assimilation_result_ocoobs_3e-5_{str(iteration).zfill(3)}.nc"),
                         time_string=time_string, plot_interval=48,
                         longitude_vector = longitude_vector,
                         latitude_vector = latitude_vector,
                         vector_z = vector_z)
            
            
            x_result = calculate_x_result(state_full_pred, weight_z)
            write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                          output_nc_name= os.path.join(args.result_folder, 
                         f"assimilation_result_ocoobs_3e-5_x{str(iteration).zfill(3)}.nc"),
                          time_string=time_string, plot_interval=48,
                          longitude_vector = longitude_vector,
                          latitude_vector = latitude_vector)
        
            torch.save(c_init_pred,os.path.join(args.result_folder, 
            f"assimilation_result_ocoobs_3e-5_cinit{str(iteration).zfill(3)}.pt"))
        
        

#%%
if( args.problem_type == "inversion" ):
    
    flux_lr = args.lr_flux

    # 注意反演的粒度
    
    # c_init = get_c_init(source_config = "geos-chem", folder_name=geoschem_folder, 
    #                     file_name = args.geoschem_co2_file, 
    #                     vertical_indexs = vertical_mapping["vertical_indexs"]  )

    # c_init = get_c_init(source_config = "init_constant", 
    #                     vertical_indexs = vertical_mapping['vertical_indexs'],
    #                     constant_value =  torch.mean(true_obs_tensor))


    # if(args.flux_type_inversion =="init_constant"):
    #     bottom_flux_pred = get_bottom_flux(source_config = "init_constant", 
    #                             time_len = u_all.shape[0])
    
    # if(args.flux_type_inversion =="from_file"):
    #     bottom_flux_pred = get_bottom_flux(source_config = "from_file", 
    #                        file_name = "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_assimilation/assimilation_result_ocoobs_3e-5_cinit019.pt")
    
    # bottom_flux_pred.requires_grad = True
    # optimizer_fluxbottom = Adam([bottom_flux_pred], lr=flux_lr)
    from global_land_mask import globe
    land_mask = torch.zeros([len(longitude_vector), len(latitude_vector)]).to(torch.float32)
    for lon_index, lon in enumerate(longitude_vector):
        for lat_index, lat in enumerate(latitude_vector):
            if(globe.is_land(lat, lon) == True):
                land_mask[lon_index, lat_index] = 1.0
                
                
    
    

    if(False):
        bottom_flux_solve = 0.0*torch.ones([1, 1, 72//2, 46//2])
        # enlarge_layer = torch.nn.ConvTranspose2d(1, 1, 2, stride=2,output_padding=(0,2))
        
        """
        做放大的层,定义为 enlarge_layer 
        """
        enlarge_layer = torch.nn.ConvTranspose2d(1, 1, 2, stride=2)
        enlarge_layer.weight = nn.Parameter(torch.ones([1,1,2,2,]))
        enlarge_layer.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        enlarge_layer.weight.requires_grad = False
        enlarge_layer.bias.requires_grad = False

    
        temp = enlarge_layer(bottom_flux_solve).squeeze(0).unsqueeze(-1)
        bottom_flux_pred_pre = torch.tile(temp,[u_all.shape[0],1,1,1])
        bottom_flux_pred = torch.einsum("tabh,ab->tabh",bottom_flux_pred_pre, land_mask)
        bottom_flux_solve.requires_grad = True
        optimizer_fluxbottom = Adam([bottom_flux_solve], lr=flux_lr)
    

    # bottom_flux_init = get_bottom_flux(source_config = "carbon_tracker", 
    #                             time_len = u_all.shape[0],
    #                             file_name = os.path.join(args.data_folder, 
    #                                                         "carbontracker/",
    #                                                         args.init_flux_file_name))


    if(True):
        # bottom_flux_solve = bottom_flux_init[0:1, :,:,:]
        bottom_flux_solve = 0.0*torch.ones([1,  72, 46,1])
        # bottom_flux_pred_pre = torch.tile(bottom_flux_solve,[u_all.shape[0],1,1,1])
        # bottom_flux_pred = torch.einsum("tabh,ab->tabh",bottom_flux_pred_pre, land_mask)

        bottom_flux_pred = torch.tile(bottom_flux_solve,[u_all.shape[0],1,1,1])
        bottom_flux_solve.requires_grad = True
        optimizer_fluxbottom = Adam([bottom_flux_solve], lr=flux_lr)

    
    
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    
    filter_op = SP_Filter_3D()
    
    loss_list = []
    flux_loss_list = []
    for iteration in range(iteration_number):
        # 需要重新赋值
        print(f"iteration:{iteration}")
        start_time = time.time()
        if(False):
            temp = enlarge_layer(bottom_flux_solve).squeeze(0).unsqueeze(-1)
            bottom_flux_pred_pre = torch.tile(temp,[u_all.shape[0],1,1,1])
            bottom_flux_pred = torch.einsum("tabh,ab->tabh",bottom_flux_pred_pre, land_mask)
        
        bottom_flux_pred = torch.tile(bottom_flux_solve,[u_all.shape[0],1,1,1])
        # bottom_flux_pred = torch.einsum("tabh,ab->tabh",bottom_flux_pred_pre, land_mask)
        
        state_all = construct_state_with_cinitfluxbottom(c_init, bottom_flux_pred, 
                                                     u_all, v_all, w_all)
        
        
        state_full_pred = ctm_simulation_3d_mixing(model, time_vector, state_all,pbl_top, 
                                                           vertical_mapping,weight_z,
                                                           if_mixing=args.if_mixing)
    
        # 变量选择浓度, 在变量和高度方向平均
        pred_obs_tensor_full = calculate_x_result(state_full_pred[:, 1:2, :,:,:], weight_z)
        if(obs_type == "full"):
            pred_obs_tensor = pred_obs_tensor_full
        if(obs_type in ["obspack","satellite"]):
            pred_obs_tensor = pred_obs_tensor_full.flatten()[index_vector]
            
        loss_mse = criterion_mse(pred_obs_tensor, true_obs_tensor)
        loss_mae = criterion_mae(pred_obs_tensor, true_obs_tensor)
        print( f"iteration:{iteration}, conct mse:{loss_mse.item():.5g}, mae:{loss_mae.item():.5g}")
        loss_list.append(loss_mae.item())

        loss_flux = criterion_mae(bottom_flux_solve, bottom_flux)
        # print(bottom_flux_solve.shape, bottom_flux.shape)
        # print(bottom_flux_solve.mean(), bottom_flux.mean())
        flux_loss_list.append(loss_flux.item())
        print(f"loss_flux:{loss_flux.item()}"  )
        
        if(loss_mae <= early_stop_value):
            print(f"early stop at iteration:{iteration}, mae:{loss_mae}")
            break
        
        if(args.if_background == "True" and args.background_weight):
            background_loss = args.background_weight*criterion_mse(bottom_flux_pred, filter_op(bottom_flux_pred) )
            print( f"iteration:{iteration}, background_loss:{background_loss.item():.3g}")
        
        if(args.if_background == "True"):
            total_loss = loss_mse + background_loss
        else:
            total_loss = loss_mse
        
        optimizer_fluxbottom.zero_grad()
        total_loss.backward()
        optimizer_fluxbottom.step()
        # print(f"elapse time:{time.time() - start_time: .3e}")
        
        if((args.if_plot_result) and ((iteration+1)%args.plot_per_iteration ==0)):
            write_carbon_netcdf_3d_geoschem(data_= state_full_pred.detach().numpy(),
                         output_nc_name= os.path.join(args.result_folder, 
                         f"inversion_result_{str(iteration).zfill(3)}.nc"),
                         time_string=time_string, plot_interval = args.plot_interval,
                         longitude_vector = longitude_vector,
                         latitude_vector = latitude_vector,
                         vector_z = vector_z)
            
            
            x_result = calculate_x_result(state_full_pred, weight_z)
            
            print(state_full_pred.shape, x_result.shape)

            write_carbon_netcdf_2d_geoschem(data_=x_result.detach().numpy(),
                          output_nc_name= os.path.join(args.result_folder, 
                         f"inversion_result_x{str(iteration).zfill(3)}.nc"),
                          time_string=time_string, plot_interval = args.plot_interval,
                          longitude_vector = longitude_vector,
                          latitude_vector = latitude_vector)
        
            torch.save(bottom_flux_pred,os.path.join(args.result_folder, 
            f"inversion_result_bottomflux{str(iteration).zfill(3)}.pt"))

            mae_score, correlation, pred_mean, true_mean, pred_abs, true_abs = calcuate_mask_score(
                bottom_flux_pred[0,:,6:-6,0].cpu().detach().numpy(), 
            bottom_flux[0,:,6:-6,0].cpu().detach().numpy(), 
            land_mask[:,6:-6].to(torch.int32).cpu().detach().numpy())

            # print( bottom_flux_pred[0,:,:,0].cpu().detach().numpy().shape,
            # bottom_flux[0,:,:,0].cpu().detach().numpy().shape,
            # land_mask.to(torch.int32).cpu().detach().numpy().shape)

            print(f"calibration: mae_score:{mae_score:.3e},correlation:{correlation:.3f}, pred_mean:{pred_mean:.3e}, true_mean:{true_mean:.3e}, \
            pred_abs:{pred_abs:.3e}, true_abs:{true_abs:.3e}"  ) 
            # print(bottom_flux_pred_pre.shape, bottom_flux.shape)


    print(loss_list)
    print(flux_loss_list)
