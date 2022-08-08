#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:15:13 2022

@author: yaoyichen
"""

import torch
from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_batch

"""
需要4份状态数据  "Met_U", "Met_V","Met_OMEGA", "Met_PBLTOPL"
"""
def construct_uvw_all_3d(merra2_folder, startswith_string, 
                         args):
    """
    读入气象数据， 生成u,v,w场， 以及边界层pbl信息
    """
    uvw_vector, file_name_list =  get_variable_Merra2_3d_batch(merra2_folder, startswith_string, 
                                               latitude_dim = 46, longitude_dim = 72,
                                               variable_list = ["Met_U", "Met_V","Met_OMEGA"] ,
                                               args = args)
    
    # print(uvw_vector.shape)
    u,v,w = uvw_vector[:,0,:,:,:], uvw_vector[:,1,:,:,:], uvw_vector[:,2,:,:,:]

    pbl_top, file_name_list = get_variable_Merra2_3d_batch(merra2_folder, startswith_string, 
                                           latitude_dim = 46, longitude_dim = 72,
                                           variable_list = ["Met_PBLTOPL",], 
                                           args = args)

    u_all = torch.tensor(u, dtype= torch.float32)
    v_all = torch.tensor(v, dtype= torch.float32)
    w_all = torch.tensor(w, dtype= torch.float32)
    pbl_top = torch.tensor(pbl_top, dtype= torch.float32)

    return  u_all, v_all, w_all, pbl_top, file_name_list


def fulfill_vertical_mapping(vertical_mapping):
    """
    构建高度方向的映射函数
    """
    result_dict = {}
    for key,value in vertical_mapping["forward"].items():     
        result_dict[value] = key
    vertical_mapping["backward"] = result_dict
    keep_vertical = list(vertical_mapping["forward"].values())
    vertical_mapping["vertical_indexs"] = keep_vertical
    return vertical_mapping


def generate_vertical_vertical_mapping_all():
    """
    产生全部高度层的  vertical_mapping

    """
    temp_dict = {}
    vertical_mapping = {}
    for i in range(47):
        temp_dict[i] = i
    vertical_mapping["forward"] = temp_dict
    return vertical_mapping



def generate_vertical_info(layer_type):
    """
    vertical_mapping = {'forward': {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 20, 7: 30, 8: 40},
     'backward': {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 20: 6, 30: 7, 40: 8},
     'vertical_indexs': [0, 2, 4, 6, 8, 10, 20, 30, 40]}
    
    forward: 从 torch vertical index 到 geos-chem vertical index 的映射
    backward:从 geos-chem vertical index 到 torch vertical index 的映射
    vertical_indexs: geos-chem 高度方向保留的层index
    """
    
    if (layer_type == "layer_9"):
        #可自行配置高度层, map key: 现有高度， value 对应到geos-chem数据的高度层
        vertical_mapping = {"forward":{0:0, 1:2, 2:4, 3:6, 4:8, 5:10,6:20,7:30,8:40}}
        
    if (layer_type == "layer_47"):
        vertical_mapping = generate_vertical_vertical_mapping_all()
        
    if (layer_type == "layer_1"):
        vertical_mapping = {"forward":{0:0}}
    
    
    vertical_mapping = fulfill_vertical_mapping(vertical_mapping)
    return vertical_mapping





 

def regenerate_vertcal_state(vector_z_origine, map_factor_origine, 
                         u_all_orgine, v_all_orgine, w_all_orgine, pbl_top_orgine,vertical_mapping):
    """
    1. 将vector_z_origine, map_factor_origine, 
         u_all_orgine, v_all_orgine, w_all_orgine 仅保留选中的高度层，
    2. 生成不同高度的权重向量 weight_z 
    3. pbl_top 代表 在 index层下，采用 turbday model做平均, 重新构建pbl_top 向量
    """
    vertical_indexs = vertical_mapping["vertical_indexs"]
    
    vector_z = vector_z_origine[vertical_indexs]
    map_factor = map_factor_origine[:,:,vertical_indexs]
    
    u_all  = u_all_orgine[:,:,:,vertical_indexs]
    v_all  = v_all_orgine[:,:,:,vertical_indexs]
    w_all  = w_all_orgine[:,:,:,vertical_indexs]
    
    
    weight_z = calculate_weight_z(vector_z)
    
    # 需要有个函数，将pbl_top_current 映射到 vertical_mapping 有的index上
    pbl_top = pbl_top_orgine.clone()
    for key,value in vertical_mapping["forward"].items():
        pbl_top[pbl_top_orgine>= value] = value
    pbl_top = pbl_top.to(torch.int)      
    print(torch.mean(pbl_top*1.0))
    return  [vector_z, map_factor, 
                             u_all, v_all, w_all], weight_z, pbl_top





def calculate_weight_z(vector_z):
    """
    由高度向量分布 vector_z ，构建权重系数 vector_z
    input:
    tensor([-99250.0000, -96250.0000, -93250.0000, -90250.0000, -87250.0000,
            -84250.0000, -58130.0039, -19250.0000,  -1450.0000])
    
    output:
    tensor([0.0225, 0.0300, 0.0300, 0.0300, 0.0300, 0.1456, 0.3250, 0.2834, 0.1035])
    """
    pressure_ratio = -vector_z/100000
    pressure_ratio_pad = torch.concat([torch.tensor([1.0]), pressure_ratio, torch.tensor([0.0])])
    left_gap = -pressure_ratio_pad[1:-1] + pressure_ratio_pad[0:-2]
    left_gap[1::] = left_gap[1::]/2.0
    right_gap = -pressure_ratio_pad[2::] + pressure_ratio_pad[1:-1]
    right_gap[0:-1] = right_gap[:-1]/2.0
    weight_z = ( left_gap + right_gap)
    return weight_z



def calculate_x_result(input_torch_tensor, weight_z):
    """
    获得高度方向加权的结果
    torch.Size([240, 5, 72, 46, 9]) -> torch.Size([240, 5, 72, 46, 1])
    """
    weighted_value = input_torch_tensor*weight_z*len(weight_z)
    
    
    return weighted_value.mean(axis = -1).unsqueeze(-1)
