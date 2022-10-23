#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 00:59:22 2022

@author: yaoyichen
"""

import h5py
import matplotlib.pyplot as plt
from datetime import datetime
"""
h5 file group https://docs.h5py.org/en/stable/high/group.html
how to read from h5py group https://stackoverflow.com/questions/56184984/read-multiple-datasets-from-same-group-in-h5-file-using-h5py

file format: oco2_L2Std[Mode]_[Orbit][ModeCounter]_[AcquisitionDate]_[ShortBuildId]_[Production DateTime] [Source].h5
"""

f = h5py.File('/Users/yaoyichen/dataset/carbon/OCO/oco3_L2StdSC_15083a_220102_B10310_220102234921.h5', 'r')
print( list(f.keys()) )

xco2 = f["RetrievalResults"]["xco2"][:]
co2_profile = f["RetrievalResults"]["co2_profile"][:]
latitude = f["RetrievalGeometry"]["retrieval_latitude"][:]
longitude = f["RetrievalGeometry"]["retrieval_longitude"][:]
time_original = f["RetrievalHeader"]["retrieval_time_string"][:]

"""
%Y-%m-%dT%H:%M:%S
原始时间格式: yyyy-mm- ddThh:mm:ss.mmmZ
"""
def convertTime(x):
    date_time_obj = datetime. strptime(x.decode("utf-8")[0:-5], '%Y-%m-%dT%H:%M:%S')
    return date_time_obj

time = [convertTime(x) for x in time_original]
altitude = f["RetrievalResults"]["vector_altitude_levels"][:]
pressure = f["RetrievalResults"]["vector_pressure_levels"][:]


plt.plot(longitude, latitude)