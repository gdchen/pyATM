#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:36:47 2022

@author: yaoyichen
"""

import h5py
import matplotlib.pyplot as plt
import  datetime
import os
import numpy as np


def plot_lonlat_onearth(longitude, latitude, value= None,dot_size = 2):
    from mpl_toolkits.basemap import Basemap
    fig = plt.gcf()
    fig.set_size_inches(10, 6.5)
    
    m = Basemap(projection='merc', \
                llcrnrlat=-70, urcrnrlat=70, \
                llcrnrlon=-180, urcrnrlon=180, \
                lat_ts=10, \
                resolution='c')
    
    m.bluemarble(scale=0.2,alpha = 0.8)   # full scale will be overkill
    m.drawcoastlines(color='white', linewidth=0.2)  # add coastlines
    x, y = m(longitude, latitude)  # transform coordinates
    
    # if( value ):
    value = 2.5*(value - np.mean(value))/(np.max(value) - np.min(value)) + 0.5
    print(value)
    
    # plt.scatter(x, y, c = value, s = dot_size, marker='o',cmap = "viridis",vmin=0, vmax=1) 
    plt.scatter(x, y, c = "white", s = dot_size, marker='o',cmap = "viridis",vmin=0, vmax=1) 
 
    plt.legend()
    # else:
    #     plt.scatter(x, y, s = 2, marker='o', color='Red') 
    
    plt.show()
    
    
    
