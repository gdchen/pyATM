#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:49:16 2022

@author: yaoyichen
"""

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from data_prepare.read_era5_uvwfc import get_c,get_f,get_uvw,get_c_point, get_c_zero,get_oco,get_variable_carbon_3d_concentration
from data_prepare.read_era5_uvwfc import get_variable_Merra2_3d_single, get_variable_Merra2_3d_batch,get_variable_Merra2_vector_single
import netCDF4 as nc
import torch


# folder_name = '/Users/yaoyichen/project_earth/data_assimilation/data/nc_file/'


# file_name = "result_simulation_3d_0222_oco_temp_smooth.nc"


# file_name = "result_simulation_3d_0222_oco_temp_upwind_2d_997.nc"
# 
# file_name = "upwind_rk4_full.nc"

# file_name = "upwind_rk4_full_mix.nc"


# experiment_mapping = {"center": {"file_name":"result_simulation_3d_0222_oco_temp_smooth.nc" ,"color":"royalblue"},
#                       "upwind": {"file_name":"result_simulation_3d_0222_oco_temp_upwind_2d_997.nc" ,"color":"green"},
#                       "upwind + vectical": {"file_name":"upwind_rk4_full.nc","color":"orange"},
#                        "upwind + vectical + pbl": {"file_name":"upwind_rk4_full_mix.nc", "color":"black"}
#                       }


experiment_mapping = {"V01": {"folder_name":"/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v01",
                                  "file_name":"forward_simulation_3d.nc" ,"color":"royalblue"},
                      
                      "V02": {"folder_name":"/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v02",
                                 "file_name":"forward_simulation_3d.nc" ,"color":"green"},
                      
                      "V03": {"folder_name":"/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v03",
                              "file_name":"forward_simulation_3d.nc","color":"orange"},
                      
                      "V04": {"folder_name":"/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v04",
                               "file_name":"forward_simulation_3d.nc", "color":"black"},
                      
                      # "V05": {"folder_name":"/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v05",
                      #         "file_name":"forward_simulation_3d.nc", "color":"black"}
                      }
#%%
# bench_folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
# bench_file_name = "co2_result_nonmixing.nc"

# bench_folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix/OutputDir/'
# bench_file_name = "co2_result_addmix.nc"

# 需要把所有的文件放到一起
# bench_folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_vdiff/OutputDir/'
# bench_file_name = "co2_result_nomixing_addmix_vdiff.nc"

bench_folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_addchem/OutputDir/'
bench_file_name = "co2_fossil.nc"

font0 = {'family'  : 'Times New Roman',
      'weight' : 'normal',
      'size'   :  30 }


font1 = {'family'  : 'Times New Roman',
      'weight' : 'normal',
      'size'   :  23 }


font2 = {'family'  : 'Times New Roman',
      'weight' : 'normal',
      'size'   :  18 }
fig, axs = plt.subplots(1,2, figsize=(22, 6.5),sharey=False)
    
def calcuate_diff_between_torch_geoschem(folder_name, experiment_mapping, bench_folder_name,bench_file_name):
    bench_folder_file_name = os.path.join(bench_folder_name, bench_file_name)
    df = nc.Dataset(bench_folder_file_name)
    c_bench_vector = torch.tensor(df["SpeciesConc_CO2"][:])
    c_bench_vector = torch.permute(c_bench_vector, [0,2,3,1]).numpy()
    

    for key, value in experiment_mapping.items():
        
        file_name = value["file_name"]
        
        folder_file_name = os.path.join(value["folder_name"], file_name)
        df = nc.Dataset(folder_file_name)
        c_vector = df["c"][:]
        
        diff = c_vector[1::,:,:,0] - c_bench_vector[1::,:,:,0]
        print(f"{np.mean(np.abs(diff))*1e6:.3f}")
        
        
        time_diff = np.mean(np.abs(diff),axis = (1,2))
        axs[0].plot(np.arange(1,336)/48.0,time_diff*1e6, label = key, color = value["color"], linewidth = 2.0)
        axs[0].set_xlabel("day", font = font1)
        axs[0].set_ylabel("Bottom CO2 MAE (ppm)",font = font1)
        axs[0].grid(linestyle = "--",c = "k")
        axs[0].legend(prop=font2, loc = "upper left" )
        axs[0].set_xlim(xmin = 0, xmax = 7.0)
        axs[0].set_ylim(ymin = 0, ymax = 1.5)
        axs[0].set_xticklabels( [0,1,2,3,4,5,6,7,], fontsize = 16)
        axs[0].set_yticklabels( [0.0, 0.2, 0.4,0.6,0.8,1.0,1.2,1.4], fontsize = 16)
        axs[0].text(-1,1.5*1.0, "(a)", font = font0)

    
        
        plt.figure(2)    
        diff = c_vector[1::,:,:,:] - c_bench_vector[1::,:,:,:]
        print(f"{np.mean(np.abs(diff))*1e6:.3f}")
        
        time_diff = np.mean(np.abs(diff),axis = (1,2,3))
        axs[1].plot(np.arange(1,336)/48.0,time_diff*1e6, label = key, color = value["color"], linewidth = 2.0)
        axs[1].set_xlabel("day", font = font1)
        axs[1].set_ylabel("XCO2 MAE (ppm)", font = font1)
        axs[1].grid(linestyle = "--",c = "k")
        axs[1].legend(prop=font2, loc = "upper left" )
        axs[1].set_xlim(xmin = 0, xmax = 7.0)
        axs[1].set_ylim(ymin = 0)
        axs[1].set_xticklabels([0,1,2,3,4,5,6,7,],fontsize = 16)
        axs[1].set_yticklabels( [0.0, 0.05, 0.10,0.15,0.20, 0.25,0.30, 0.35,0.40], fontsize = 16)
        axs[1].text(-1,0.4*1.0, "(b)", font = font0)
        # axs[1].title("difference of XCO2")



calcuate_diff_between_torch_geoschem(folder_name, experiment_mapping, bench_folder_name,bench_file_name)

#%%

def calcuate_diff_between_geoschem(bench_folder_name1, bench_file_name1,bench_folder_name2, bench_file_name2):
    """
    获得2份 geos-chem结果之间的误差
    """
    bench_folder_file_name = os.path.join(bench_folder_name1, bench_file_name1)
    df = nc.Dataset(bench_folder_file_name)
    c_bench_vector = torch.tensor(df["SpeciesConc_CO2"][:])
    c_bench_vector1 = torch.permute(c_bench_vector, [0,2,3,1]).numpy()
    
    bench_folder_file_name = os.path.join(bench_folder_name2, bench_file_name2)
    df = nc.Dataset(bench_folder_file_name)
    c_bench_vector = torch.tensor(df["SpeciesConc_CO2"][:])
    c_bench_vector2 = torch.permute(c_bench_vector, [0,2,3,1]).numpy()
    
    diff = c_bench_vector1[1::,:,:,:] - c_bench_vector2[1::,:,:,:]
    print(f"{np.mean(np.abs(diff))*1e6:.3f}")
    
    time_diff = np.mean(np.abs(diff),axis = (1,2,3))
    plt.plot(np.arange(1,432)/72.0,time_diff*1e6, label = "diff", color = value["color"])
    plt.xlabel("day")
    plt.ylabel("mae (ppm)")
    plt.legend()
    
    return 

# bench_folder_name1 = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_vdiff/OutputDir/'
# bench_file_name1 = "co2_result_nomixing_addmix_vdiff.nc"

# bench_folder_name2 = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix/OutputDir/'
# bench_file_name2 = "co2_result_addmix.nc"
# bench_folder_name1 = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addmix_addchem/OutputDir/'
# bench_file_name1 = "co2_result_nomixing_addmix_addchem.nc"


calcuate_diff_between_geoschem(bench_folder_name1,bench_file_name1, bench_folder_name2, bench_file_name2 )


#%%
folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare/OutputDir/'
startswith_string = "GEOSChem.StateMet.201907"
variable_list = ["c", "Met_V"] 
u, v = get_variable_Merra2_3d_single(folder_name, startswith_string, latitude_dim = 46, longitude_dim = 72,variable_list = variable_list)



#%%
file_name = "GEOSChem.StateMet.20190701_1940z.nc4"

folder_file_name = os.path.join(folder_name, file_name)




longitude, latitude = get_variable_Merra2_vector_single(folder_name,file_name, variable_list = ["lon", "lat"] )

        # latitude = np.asarray(df.variables[varible][:])
# level = df.variables["level"][:]

# longitude = np.asarray(df.variables["longitude"][:])
# latitude = np.asarray(df.variables["latitude"][:])
folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare/OutputDir/'
file_name = "GEOSChem.SpeciesConc.20190701_0000z.nc4"
variable_list = ["SpeciesConc_CO2"]

[tt] = get_variable_Merra2_3d_single(folder_name, file_name, latitude_dim = 46, longitude_dim = 72, variable_list = variable_list)



folder_name = '/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing/OutputDir/'
file_name = "GEOSChem.StateMet.20190701_0000z.nc4"
variable_list = ["Met_PBLTOPL"]
[pbl_height] = get_variable_Merra2_vector_single(folder_name, file_name,  variable_list = variable_list)