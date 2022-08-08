#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:05:41 2022

@author: yaoyichen
"""
#%%
import wget, tarfile
import os
import datetime
import pandas as pd
start = datetime.datetime.strptime("2017-08-01", "%Y-%m-%d")
end = datetime.datetime.strptime("2019-03-28", "%Y-%m-%d")
date_generated = pd.date_range(start, end)
date_list = list(date_generated.strftime("%Y-%m-%d"))

 
# 网络地址
date_str = '2000-01-01'

"""
XCO2
"""
# for date_str in date_list:
#     DATA_URL = f'https://gml.noaa.gov/aftp//products/carbontracker/co2/CT2019B/molefractions/xCO2_1330LST/CT2019B.xCO2_1330_glb3x2_{date_str}.nc'
    
#     out_folder_name = "/Users/yaoyichen/dataset/carbon/carbonTracker/xco2/"
#     out_file_name = f'CT2019B.xCO2_1330_glb3x2_{date_str}.nc'
#     wget.download(DATA_URL, out=os.path.join(out_folder_name, out_file_name) )



# for date_str in date_list:
date_str = '2018-07-01'
DATA_URL = f'https://gml.noaa.gov/aftp//products/carbontracker/co2/CT2019B/molefractions/co2_total/CT2019B.molefrac_glb3x2_{date_str}.nc'

out_folder_name = "/Users/yaoyichen/dataset/carbon/carbonTracker/total_co2/"
out_file_name = f'CT2019B.molefrac_glb3x2_{date_str}.nc'
wget.download(DATA_URL, out=os.path.join(out_folder_name, out_file_name) )





