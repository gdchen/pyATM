#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:39:19 2022

@author: yaoyichen
"""

import datetime
import numpy as np



"""
%Y-%m-%dT%H:%M:%S
原始时间格式: yyyy-mm- ddThh:mm:ss.mmmZ
"""
def convertTime(x):
    date_time_obj = datetime.strptime(x.decode("utf-8")[0:-5], '%Y-%m-%dT%H:%M:%S')
    return date_time_obj



def plus_seconds(x):
    start_time = datetime.datetime(1970, 1, 1, 0, 0, 0)
    result_time = start_time + datetime.timedelta(seconds=x)
    return result_time