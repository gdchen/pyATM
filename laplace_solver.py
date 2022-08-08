import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from dynamic_model.differ_module import Order2_Diff2_Unstructure, Order2_Diff1_Unstructure
import os
from tools.model_helper import ModelUtils
from tools.file_helper import FileUtils
from tools.plot_tools import plot_2d

dx = 0.04
dy = 0.04
vector_x = torch.arange(0,  50*dx, dx, dtype=torch.float32)
vector_y = torch.arange(0,  25 * dy, dy, dtype=torch.float32)
grid_x, grid_y = torch.meshgrid(vector_x, vector_y)

grid_info = (dx, dy, grid_x, grid_y, vector_x, vector_y)


def get_omega():
    x0, y0 = 1.0, 0.5
    R0 = 0.15
    r = torch.sqrt((grid_x - x0) ** 2 + (grid_y - y0) ** 2)
    omega = torch.exp(-(r/R0)**2)
    return omega


omega = get_omega()
omega = omega.unsqueeze(0)


phi = torch.zeros(omega.shape)
phi.requires_grad = True


class Get_Laplace(nn.Module):
    def __init__(self, x_vector, y_vector):
        super().__init__()
        self.diff_x = Order2_Diff2_Unstructure(
            x_vector, total_dim=2, diff_dim=1)
        self.diff_y = Order2_Diff2_Unstructure(
            y_vector, total_dim=2, diff_dim=2)

        self.grad_x = Order2_Diff1_Unstructure(
            x_vector, total_dim=2, diff_dim=1)
        self.grad_y = Order2_Diff1_Unstructure(
            y_vector, total_dim=2, diff_dim=2)

    def forward(self, state):
        return self.diff_x(state) + self.diff_y(state)

    def getuv_from_phi(self, phi):
        return self.grad_y(phi), -self.grad_x(phi)


optimizer = Adam([phi], lr=3e-4)


# optimizer = torch.optim.LBFGS(
#     [phi],
#     lr=1.0,
#     max_iter=50000,
#     max_eval=50000,
#     history_size=50,
#     tolerance_grad=1e-5,
#     tolerance_change=1.0 * np.finfo(float).eps,
#     line_search_fn="strong_wolfe"       # can be "strong_wolfe"
# )


criterion = nn.MSELoss()

model = Get_Laplace(vector_x, vector_y)


for i in range(300):
    la_phi = model(phi)
    optimizer.zero_grad()
    loss = criterion(la_phi, omega)
    loss.backward()
    optimizer.step()
    print(loss)

with torch.no_grad():
    u, v = model.getuv_from_phi(phi)
    u = u + 1.0

# 为了画图用
foldername = os.path.join(
    "./results/laplace/")
FileUtils.makedir(foldername)


plot_2d(omega.squeeze().detach(), foldername, "result_omega.png")
plot_2d(phi.squeeze().detach(), foldername, "result_phi.png")

plot_2d(u.squeeze().detach(), foldername, "result_u.png")
plot_2d(v.squeeze().detach(), foldername, "result_v.png")
