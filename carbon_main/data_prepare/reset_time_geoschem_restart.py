#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:59:16 2022

@author: yaoyichen
"""
#%%

import netCDF4 as nc
f = nc.Dataset("/Users/yaoyichen/project_earth/gc_experiment/gc_merra2_CO2_compare_nomixing_addconvection_30min/GEOSChem.Restart.20181201_0000z.nc4","r+")
f["time"].units = 'hours since 2018-12-01 00:00:00'
f.close()

