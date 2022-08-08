#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:48:00 2022

@author: yaoyichen
"""
import torch
import numpy as np
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period
import matplotlib.pyplot as plt
import datetime
from dynamic_model.differ_module import Order2_Diff1_Unstructure, Order2_Diff1_Unstructure_Period
from dynamic_model.differ_module import Order2_Diff2_Unstructure, Order2_Diff2_Unstructure_Period
from dynamic_model.differ_module import Order2_Diff1_Unstructure_Perioid_Upwind,Order2_Diff1_Unstructure_Upwind



x_np = np.arange(100)/100*np.pi*2
x = torch.tensor(x_np).to(torch.float32)
x_vector = x.unsqueeze(0).unsqueeze(-1)
y = torch.sin(3*x).unsqueeze(0).unsqueeze(-1)

diff1_x_upwind =  Order2_Diff1_Unstructure_Upwind(
        x, total_dim=2, diff_dim=1)

diff1_x =  Order2_Diff1_Unstructure(
        x, total_dim=2, diff_dim=1)


x_diff_upwind = diff1_x_upwind(y,-y)
x_diff = diff1_x(y)


plt.plot(x_vector.squeeze(), y.squeeze())
plt.plot(x_vector.squeeze(), x_diff_upwind.squeeze())
plt.plot(x_vector.squeeze(), x_diff.squeeze())


#%%
diff1_x_upwind =  Order2_Diff1_Unstructure_Upwind(
        vector_x, total_dim=2, diff_dim=1)

diff1_x =  Order2_Diff1_Unstructure(
        vector_x, total_dim=2, diff_dim=1)

input_u = X[0,:,:]
x_diff_upwind = diff1_x_upwind(input_,input_u)
x_diff = diff1_x(input_)
