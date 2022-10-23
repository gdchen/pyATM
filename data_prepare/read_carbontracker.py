#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:16:28 2022

@author: yaoyichen
"""

# 读入carbon tracker 数据， 并能够差值到现有的网格上

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset,num2date,date2num
import torch
import cv2
import matplotlib.pyplot as plt


#%%

def reshape_ct_flux(input_file, output_file):
    """
    输入 (1, 180, 360)
    输出  (72, 46)
    """
    
    df = nc.Dataset(input_file)
    
    bio = df["bio_flux_opt"][:]
    fossil = df["fossil_flux_imp"][:]
    ocean = df["ocn_flux_opt"][:]
    fire = df["fire_flux_imp"][:]
    total = bio + fossil + ocean 
    # + fire

    
    bio_sum = np.sum(np.abs(bio) )
    fossil_sum = np.sum(np.abs(fossil) )
    ocean_sum = np.sum(np.abs(ocean) )
    fire_sum = np.sum(np.abs(fire) )
    total_sum = bio_sum + fossil_sum + ocean_sum 
    print(f"bio percentrage:{bio_sum/total_sum}" )
    print(f"fossil percentrage:{fossil_sum/total_sum}" )
    print(f"ocean percentrage:{ocean_sum/total_sum}" )
    print(f"fire percentrage:{fire_sum/total_sum}" )
    
    
    
    #cv2.INTER_CUBI
    total_flux_reshape = cv2.resize(total[0,:,:], dsize=(72,46), interpolation=cv2.INTER_LINEAR)
    print(np.max(total), np.max(total_flux_reshape))
    print(np.mean(total),np.mean(total_flux_reshape))
    print(np.min(total),np.min(total_flux_reshape))
    
    total_flux_reshape_permute = total_flux_reshape.transpose(1,0)
    np.save(output_file,total_flux_reshape_permute[np.newaxis,:])
    # print("#"*20)
    print(total.shape, total_flux_reshape_permute.shape)
    
    return total, total_flux_reshape



def generate_np_file(input_file, output_file):
    # input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.201807.nc"
    # output_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/CT2019B.flux1x1.201807_reshape.npy"
    total, total_flux_reshape = reshape_ct_flux(input_file, output_file)
    return 1
    
    

# input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.201807.nc"


# input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/daily_flux/CT2019B.flux1x1.20180701.nc"

def generate_nc_file(input_file, output_file):
    nc_in = nc.Dataset(input_file)
    bio = nc_in["bio_flux_opt"][:]
    fossil = nc_in["fossil_flux_imp"][:]
    ocean = nc_in["ocn_flux_opt"][:]
    fire = nc_in["fire_flux_imp"][:]

    total = bio + fossil + ocean 


    result_shape = np.empty([1, 46, 72])
    for i in range(1):
        total_flux_reshape = cv2.resize(total[i,:,:], dsize=(72,46), interpolation=cv2.INTER_LINEAR)
        result_shape[i,:,:] = total_flux_reshape
    # total_flux_reshape = total_flux_reshape[np.newaxis,:,:]

    ncout = Dataset(output_file, 'w', 'NETCDF4')
    ncout.createDimension('longitude', 72)
    ncout.createDimension('latitude', 46)
    ncout.createDimension('time', None)


    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    timevar.setncattr('units', "minutes since 2000-01-01 00:00:00")

    fluxvar = ncout.createVariable("f", 'float32',
                                ('time', 'latitude', 'longitude'))

    lonvar[:] = np.arange(-180, 180, 5)
    latvar[:] = np.arange(-90,  94,  4)
    fluxvar[:] = result_shape
    timevar[:] = nc_in["time"][:]*1440

    ncout.close()
    return 1



def pipeline():
    """
    总流程
    """
    for i in range(1,13):
        input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.2017" + str(i).zfill(2) + ".nc"
        output_np_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux_np/CT2019B.flux1x1.2017" + str(i).zfill(2) + "_reshape.npy"
        output_nc_file =  "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux_nc/CT2019B.flux1x1.2017" + str(i).zfill(2) + "_reshape.nc"
        generate_np_file(input_file, output_np_file)
        generate_nc_file(input_file, output_nc_file)





def concate_all_nc():
    # input_file = 
    # output_file = 
    f_var_np = np.empty([12, 46, 72])
    t_var_np = []
    output_file =  "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux_nc/CT2019B.flux1x1.2017_merge.nc"

    for i in range(1,13):
        input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux_nc/CT2019B.flux1x1.2017" + str(i).zfill(2) + "_reshape.nc"
        nc_in =  nc.Dataset(input_file)
        t_var_np.append( nc_in["time"][:])
        f_var_np[i-1,:,:] = nc_in["f"][:]

    print(len(t_var_np), f_var_np.shape)

    ncout = Dataset(output_file, 'w', 'NETCDF4')
    ncout.createDimension('longitude', 72)
    ncout.createDimension('latitude', 46)
    ncout.createDimension('time', None)


    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    timevar.setncattr('units', "minutes since 2000-01-01 00:00:00")

    fluxvar = ncout.createVariable("f", 'float32',
                                ('time', 'latitude', 'longitude'))

    lonvar[:] = np.arange(-180, 180, 5)
    latvar[:] = np.arange(-90,  94,  4)
    fluxvar[:] = f_var_np
    timevar[:] = t_var_np

    ncout.close()
    return 1



# pipeline()
# concate_all_nc()
#%%

data_ = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.2018-monthly.nc"
df = nc.Dataset(data_) 
bio = df["bio_flux_opt"][:]

fossil = df["fossil_flux_imp"][:]
ocean = df["ocn_flux_opt"][:]
fire = df["fire_flux_imp"][:]
total = bio + fossil + ocean 

#%%
u, s, vh = np.linalg.svd(fossil[0,:,:], full_matrices=True)


tt = u[:,0:10].data @ np.diag(s[0:10]) @ vh[0:10,:].data

# u3, s3, vh3 = np.linalg.svd(fossil, full_matrices=False)


# import numpy as np
# from sklearn.decomposition import PCA

# pca_784 = PCA(n_components=784)
# pca_784.fit(fossil)