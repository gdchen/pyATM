#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 00:21:05 2022

@author: yaoyichen
"""


import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from mpl_toolkits.basemap import Basemap
from read_utils import plus_seconds


file_name = "/Users/yaoyichen/dataset/carbon/ObsPack/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18/data/daily/obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18.20201030.nc"
df = nc.Dataset(file_name)

longitude = np.asarray(df.variables["longitude"][:])
latitude = np.asarray(df.variables["latitude"][:])
time_original = np.asarray(df.variables["time"][:])

# def plus_seconds(x):
#     start_time = datetime(1970, 1, 1, 0, 0, 0)
#     result_time = start_time + timedelta(seconds=x)
#     return result_time

time = [plus_seconds(int(x)) for x in time_original]
"""
测量距离地面的高度 
"""
intake_height = np.asarray(df.variables["intake_height"][:])

model_sample_window_start = np.asarray(df.variables["model_sample_window_start"][:])
model_sample_window_end =  np.asarray(df.variables["model_sample_window_end"][:])
start_time = [plus_seconds(int(x)) for x in model_sample_window_start]
end_time  = [plus_seconds(int(x)) for x in model_sample_window_end]

value  = np.asarray(df.variables["value"][:])


# import numpy as np
# import matplotlib.pyplot as plt
# exit()

# make up some data for scatter plot
# lats = np.random.randint(-75, 75, size=20)
# lons = np.random.randint(-179, 179, size=20)

"""
下面纯粹就是画图的程序了
"""

lats = latitude
lons = longitude

fig = plt.gcf()
fig.set_size_inches(8, 6.5)

m = Basemap(projection='merc', \
            llcrnrlat=-80, urcrnrlat=80, \
            llcrnrlon=-180, urcrnrlon=180, \
            lat_ts=20, \
            resolution='c')
    
# 美国
m = Basemap(projection='merc', \
            llcrnrlat=15, urcrnrlat=65, \
            llcrnrlon=-160, urcrnrlon=-40, \
            lat_ts=20, \
            resolution='c')
# 
m = Basemap(projection='merc', \
            llcrnrlat=35, urcrnrlat=65, \
            llcrnrlon=-10, urcrnrlon=30, \
            lat_ts=20, \
            resolution='c')

m.bluemarble(scale=0.2)   # full scale will be overkill
m.drawcoastlines(color='white', linewidth=0.2)  # add coastlines

x, y = m(lons, lats)  # transform coordinates
plt.scatter(x, y, s = 5, marker='o', color='Red') 

plt.show()