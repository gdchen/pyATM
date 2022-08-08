#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:55:27 2022

@author: yaoyichen
"""

import numpy as np
import numpy.ma as ma 

def mask_data(input_, mask_):
    masked_input = ma.masked_where(mask_ == 1, input_)
    return masked_input.data.flatten()[masked_input.mask.flatten()]
    
    

def calcuate_mask_score(input_a, input_b, mask_tensor):

    a_masked = mask_data(input_a, mask_tensor)
    b_masked = mask_data(input_b, mask_tensor)
    
    mean_error = np.mean(np.abs(a_masked - b_masked))
    corr_score = np.corrcoef(a_masked, b_masked)[1,0]
    input_a_mean = np.mean(input_a)
    input_b_mean = np.mean(input_b)

    input_a_abs = np.mean(np.abs(input_a))
    input_b_abs = np.mean(np.abs(input_b))
    # print("#"*20)
    # print(a_masked.shape)
    # print(b_masked.shape)
    
    return mean_error, corr_score, input_a_mean, input_b_mean,input_a_abs, input_b_abs
    


