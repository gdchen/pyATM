
# %%
import os
import torch
import torch.nn as nn
import time
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from neural_model.differ_module import Order2_Diff1_Unstructure, Order2_Diff1, Order2_Diff2_Unstructure, Order2_Diff2
from neural_model.differ_module import Order2_Diff1_Unstructure_Period, Order2_Diff2_Unstructure_Period

import numpy as np
import scipy.stats as st


class Args:
    adjoint = False


args = Args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


dx = 0.01
x = torch.arange(0, 100 * dx, dx, dtype=torch.float32)
u3 = torch.sin(
    10 * 3.14159 * x).unsqueeze(0).unsqueeze(0)
u = torch.sin(
    10 * 3.14159 * x).unsqueeze(0).unsqueeze(0).unsqueeze(0)


diff1 = Order2_Diff2(0.01, half_padding=1)
time1 = time.time()
dudx = diff1(u3)
print('uniform     time:{:.8f}'.format(time.time() - time1))

diff1_scheme_uniform = Order2_Diff2_Unstructure_Period(
    x, total_dim=3, diff_dim=3)
time1 = time.time()
dudx_uniform = diff1_scheme_uniform(u)
print('unstructure time:{:.8f}'.format(time.time() - time1))

fig, ax = fig, ax = plt.subplots(figsize=(6, 4))

plt.plot(x, dudx.squeeze().cpu().detach().numpy(),
         '-o', label="periodic", markersize=2)


plt.plot(x, dudx_uniform.squeeze().cpu().detach().numpy(),
         '-o', label="scheme_uniform", markersize=2)


# %%
z = 1-torch.sin(np.pi / 2.0 * x)
diff1_scheme_nonuniform = Order2_Diff2_Unstructure(z, total_dim=3, diff_dim=3)
uz = torch.sin(
    10 * 3.14159 * z).unsqueeze(0).unsqueeze(0).unsqueeze(0)


dudx_nonuniform = diff1_scheme_nonuniform(uz)
plt.plot(z, dudx_nonuniform.squeeze().cpu().detach().numpy(),
         '-o', label="scheme_nonuniform", markersize=2)
plt.legend()
plt.savefig("unstructure.png")

print(torch.mean(torch.abs(dudx_uniform - dudx)))
