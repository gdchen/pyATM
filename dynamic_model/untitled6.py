#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:09:36 2022

@author: yaoyichen
"""
# 获得线性平面
#%%
import torch

tt = X[2,:,:]

def fft_res_decompose_field(input_, long_mode = 8, lati_mode = 8):
    """
    将field分解到 fft系数 
    """
    lat_mean = input_.mean(dim = 0)
    input_res = input_ - lat_mean
    input_res_fft = torch.fft.fft2(input_res)

    sp0 = input_res_fft[0:long_mode + 1, 0:lati_mode + 1]
    sp1 = input_res_fft[-long_mode::, 1:lati_mode + 1]
    
    return lat_mean, sp0, sp1


def fft_decompose_field(input_, long_mode = 8, lati_mode = 8):
    """
    将field分解到 fft系数 
    """
    input_res_fft = torch.fft.fft2(input_)

    sp0 = input_res_fft[0:long_mode + 1, 0:lati_mode + 1]
    sp1 = input_res_fft[-long_mode::, 1:lati_mode + 1]
    
    return sp0, sp1


def flatten_spectral_coefficient(sp0, sp1):
    """
    将谱系数拍平到一维
    """
    result = torch.concat([sp0.flatten() , sp1.flatten()])
    return result


def recover_spectral_coefficient(spectral_cof, long_mode = 8, lati_mode = 8):
    sp0 = spectral_cof[0:(long_mode+1)* (lati_mode+1 )].reshape([(long_mode+1),(lati_mode+1) ])
    sp1 = spectral_cof[(long_mode+1) * (lati_mode+1):: ].reshape([(long_mode),(lati_mode) ])
    return sp0, sp1


def fft_res_recover_field(lat_mean, sp0, sp1, long_mode = 8, lati_mode = 8, long_dim = 64, lati_dim= 32):
    """
    将fft系数恢复到 原始的空间
    """
    result = torch.empty([long_dim, lati_dim], dtype = sp0.dtype)
    
    result[0:long_mode + 1, 0:lati_mode + 1] = sp0
    result[0, -lati_mode::] = torch.conj(torch.flip(sp0[0, 1 :lati_mode + 1], dims = (0,)))
    result[-long_mode::, 0 ] = torch.conj(torch.flip(sp0[1:long_mode + 1,0], dims = (0,)) )
    result[-long_mode::, -lati_mode::] = torch.conj(torch.flip(sp0[1 :long_mode + 1, 1 :lati_mode + 1],dims = (0,1)))
    
    result[-long_mode::, 1:lati_mode + 1] = sp1
    result[1:(long_mode + 1), -lati_mode::] = torch.conj(torch.flip(sp1,dims = (0,1)))
    
    result_ifft = torch.fft.ifft2(result).to(torch.float32) + lat_mean
    
    return result_ifft


def fft_recover_field(sp0_value, sp1_value, long_mode = 8, lati_mode = 8, long_dim = 64, lati_dim= 32):
    """
    将fft系数恢复到 原始的空间
    """
    result_temp = torch.zeros([long_dim, lati_dim], dtype = sp0_value.dtype)
    
    result_temp[0:long_mode + 1, 0:lati_mode + 1] = sp0_value
    result_temp[0, -lati_mode::] = torch.conj(torch.flip(sp0_value[0, 1 :lati_mode + 1], dims = (0,)))
    result_temp[-long_mode::, 0 ] = torch.conj(torch.flip(sp0_value[1:long_mode + 1,0], dims = (0,)) )
    result_temp[-long_mode::, -lati_mode::] = torch.conj(torch.flip(sp0_value[1 :long_mode + 1, 1 :lati_mode + 1],dims = (0,1)))
    
    result_temp[-long_mode::, 1:lati_mode + 1] = sp1_value
    result_temp[1:(long_mode + 1), -lati_mode::] = torch.conj(torch.flip(sp1_value,dims = (0,1)))
    
    # print(result_temp.sum())
    result_ifft = torch.fft.ifft2(result_temp).to(torch.float64) 
    # print(result_temp)
    
    
    return result_ifft



long_mode = 8
lati_mode = 5
long_dim = 64
lati_dim= 32


# tt4 = torch.fft.ifft2(tt3).to(torch.float32)

nx_h = 32

def long2lat(input_, ):
    """
    经纬度互换
    """
    long_dim , lati_dim = input_.shape[0],input_.shape[1]
    nx_h = long_dim//2
    back_flip = torch.flip(variable[nx_h::, :], dims=(1,))
    result = torch.cat([input_[0:nx_h, ::], back_flip[:,:]], dim=1)
    print(back_flip, result)
    
    return result 


def long2lat_full(input_, ):
    """
    经纬度互换
    """
    long_dim , lati_dim = input_.shape[0],input_.shape[1]
    back_flip = torch.flip(input_[:, :], dims=(1,))
    back_flip_roll = torch.roll(back_flip,shifts = [long_dim//2],dims = (0,))
    
    result = torch.cat([input_, back_flip_roll], dim=1)
    
    return result 


    # d = torch.cat([b[:, 1:-1], variable[0:nx_h, ::], ], dim=1)

#%%

feature = X[2,:,:]
label = y[2,:,:]


lat_mean, sp0, sp1 = fft_res_decompose_field(tt, long_mode, lati_mode)
spectral_cof = flatten_spectral_coefficient(sp0, sp1)

sp0_recover, sp1_recover = recover_spectral_coefficient(spectral_cof, long_mode, lati_mode)

result =  fft_res_recover_field(lat_mean, sp0_recover, sp1_recover, long_mode , lati_mode,long_dim , lati_dim)


loss  = torch.mean(torch.abs(result - tt))
print(loss)


# 系数需要归一化


#%%
long_mode = 16
lati_mode = 24

# feature_res = feature_res
def calculate_error(feature):
    feature_res = feature 
    feature_res = feature_res
    feature_full = long2lat_full(feature_res)
    sp0_, sp1_ = fft_decompose_field(feature_full,long_mode, lati_mode )
    """
    这里给个小扰动    
    """
    sp0_[1,1] = sp0_[2,2]*0.8 + 0.2*sp0_[2,2]
    
    
    result_ttt = fft_recover_field(sp0_, sp1_, long_mode, lati_mode, long_dim = 64, lati_dim= 64)
    loss  = torch.mean(torch.abs(result_ttt[:,0:32] - feature_full[:,0:32]))
    print(loss)
    return sp0_, sp1_
    # print(loss, torch.std(result), torch.std(feature_res),)

for i in range(100):
    sp0, sp1 = calculate_error(feature)
    # print(sp0.mean(), sp1.mean())
# plt.figure(0)
# plt.imshow(feature_full)

# plt.figure(1)
# plt.imshow(result)


plt.plot(sp0.flatten().abs())
plt.plot(sp1.flatten().abs())

#%%
def mask_data(input_, mask_):
    masked_input = ma.masked_where(mask_ == 1, input_)
    return masked_input.data.flatten()[masked_input.mask.flatten()]
    
    

def calcuate_mask_score(input_a, input_b, mask_tensor):

    a_masked = mask_data(input_a, mask_tensor)
    b_masked = mask_data(input_b, mask)
    
    mean_error = np.mean(np.abs(a_masked - b_masked))
    corr_score = np.corrcoef(a_masked, b_masked)[1,0]
    input_a_mean = np.mean(input_a)
    input_b_mean = np.mean(input_b)
    
    return mean_error, corr_score, input_a_mean, input_b_mean
    



import numpy.ma as ma 

a = np.random.rand(72,46)
b = np.random.rand(72,46)
mask = np.random.randint(low = 0, high = 3, size = (72,46))


mean_error, corr_score, mean_a, mean_b = calcuate_mask_score(a, b, mask)
print(mean_error, corr_score, mean_a, mean_b) 