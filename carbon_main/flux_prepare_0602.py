#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:45:03 2022

@author: yaoyichen
"""
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset,num2date,date2num
import torch
import torch.nn as nn



#%% part 1
# 需要有一个加载 region_layer_torch 和 ecosystems_layer_torch 的模块
def generate_ct_region_ecosystem_torch(input_file):
    nc_in =  nc.Dataset(input_file)

    # 分为 23类
    transcom_regions_data = nc_in["transcom_regions"][:]

    # 分为 19类
    land_ecosystems_data = nc_in["land_ecosystems"][:]

    region_layer = np.zeros([23, 360, 180])
    for i in range(1, 23 + 1):
        region_temp = np.float32(transcom_regions_data == i)
        region_layer[i-1, :, : ] = np.transpose(region_temp,[1,0])
    region_layer_torch =  torch.tensor(region_layer,dtype = torch.float32)
    
    ecosystems_layer = np.zeros([19,360, 180])
    for i in range(1, 19 + 1):
        ecosystems_temp = np.float32(land_ecosystems_data == i)
        ecosystems_layer[i-1, :, : ] = np.transpose(ecosystems_temp,[1,0])
    ecosystems_layer_torch = torch.tensor(ecosystems_layer, dtype = torch.float32)
    
    return region_layer_torch, ecosystems_layer_torch


# 读入陆地生态系统数据分布   
input_file = "/Users/yaoyichen/dataset/carbon/regions.nc"

region_layer_torch, ecosystems_layer_torch = generate_ct_region_ecosystem_torch(input_file)
region_coefficient = torch.randn(23, dtype = torch.float32)
ecosystems_coefficient = torch.randn(19, dtype = torch.float32)

def encompose_ecosystem_flux(region_layer_torch, region_coefficient, ecosystems_layer_torch, ecosystems_coefficient):
    """
    输出结果 360*180
    """
    region_merge = torch.einsum("lxy,l->xy", region_layer_torch, region_coefficient )
    ecosystems_merge = torch.einsum("lxy,l->xy", ecosystems_layer_torch, ecosystems_coefficient )
    return region_merge + ecosystems_merge

print(f"ecosystems_merge.shape:{ecosystems_merge.shape}")
plt.imshow(region_merge + ecosystems_merge) 

#%% part_2
# 读入carbontracker的数据, 能够提前产生分解的模块

def get_carbontracker_data(input_file):
    """
    读取 carbontracker数据
    """
    df =  nc.Dataset(input_file)
    bio = df["bio_flux_opt"][:]
    fossil = df["fossil_flux_imp"][:]
    ocean = df["ocn_flux_opt"][:]
    fire = df["fire_flux_imp"][:]
    total = bio + fossil + ocean 
    total = total.data[:,:,:]
    return total


def decompose_carbontracker_flux(input_file, mode_number = 15):
    """
    分解carbon_tracker的结果
    """
    carbontracker_data_temp = get_carbontracker_data(input_file)
    carbontracker_data_torch =  torch.tensor(np.transpose(carbontracker_data_temp,[0,2,1]), dtype = torch.float32)
    U, S, Vh = torch.linalg.svd(carbontracker_data_torch, full_matrices=False)
    return U[:,:,0:mode_number], S[:,0:mode_number], Vh[:,0:mode_number,:]


def fetch_carbontracker_flux(input_file, index = 0):
    carbontracker_data_temp = get_carbontracker_data(input_file)
    carbontracker_data_torch =  torch.tensor(np.transpose(carbontracker_data_temp,[0,2,1]), dtype = torch.float32)

    return carbontracker_data_torch[index, :,:]
    

def encompose_carbontracker_flux(u_, s_, v_):
    """
    将u_, s_, v_ 组合成最终的结果
    """
    inverse_result = torch.mean(u_ @ torch.diag_embed(s_) @ v_, dim = 0)
    return inverse_result

#%%
input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.2017-monthly.nc"
u, s ,v = decompose_carbontracker_flux(input_file, mode_number = 20)

# s的维度 12 * 20
inverse_result = encompose_carbontracker_flux(u, s, v)
print(f"inverse_result.shape:{inverse_result.shape}")
plt.imshow(inverse_result)


#%%



class ConvModule(nn.Module):
    """
    1,360,180 -> 1, 72, 46
    """
    def __init__(self, feature_channel = 1, output_channel = 1):
        super(ConvModule, self).__init__()
        self.conv_layer = nn.Conv2d( feature_channel, output_channel, 
                                    kernel_size=(5,4), stride = (5,4), 
                                    padding=2, bias=True)
    def forward(self, x):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 1)
        # print(x.shape)
        result = self.conv_layer(x)
        return result
    

original_flux = torch.randn(1,360,180)
convModule = ConvModule(feature_channel = 1, output_channel = 1 )
result = convModule(original_flux)
print(f"result.shape:{result.shape}")

        
#%%

carbontracker_mode_coefficient = s
region_coefficient = torch.randn(23, dtype = torch.float32) 
ecosystems_coefficient = torch.randn(19, dtype = torch.float32)
original_flux = torch.zeros(360,180, dtype = torch.float32)

total_flux = 0.3 * encompose_carbontracker_flux(u, carbontracker_mode_coefficient, v) \
    + 0.3* encompose_ecosystem_flux(region_layer_torch, region_coefficient, ecosystems_layer_torch, ecosystems_coefficient) \
    + 0.4* original_flux 
     
convModule = ConvModule(feature_channel = 1, output_channel = 1 )
total_flux_geoschem = convModule(total_flux)


#%%
"""
如何获得regional flux 的先验信息。 
"""
from torch.optim import Adam

criterion_mse = torch.nn.MSELoss()

region_coefficient = torch.zeros(23, dtype = torch.float32) 
ecosystems_coefficient = torch.zeros(19, dtype = torch.float32)
region_coefficient.requires_grad = True
ecosystems_coefficient.requires_grad = True

# optimizer1 = Adam([region_coefficient], lr = 0.001)
optimizer2 = Adam([ecosystems_coefficient], lr = 0.0001)

input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.2017-monthly.nc"
true_result = fetch_carbontracker_flux(input_file, index = 6)

for i in range(100):
    predict_result = encompose_ecosystem_flux(region_layer_torch, region_coefficient, ecosystems_layer_torch, ecosystems_coefficient)
    loss = criterion_mse(true_result,  predict_result )
    # print(loss)import pandas as pd
import pandas_ta as ta

df = pd.DataFrame() # Empty DataFrame


    print( torch.mean(torch.abs(true_result -predict_result) ))
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()

plt.figure(0)
plt.imshow(true_result.detach().numpy())
plt.figure(1)
plt.imshow(predict_result.detach().numpy())





