

import torch
import torch.nn as nn


class Order2_Diff1(nn.Module):
    """
    first order diff, with accuracy order 2
    """

    def __init__(self, dx, half_padding=1):
        super(Order2_Diff1, self).__init__()
        # circular condition, should be changed at the boundary
        self.half_padding = half_padding
        self.kernel_size = 3
        self.conv_layer = nn.Conv1d(
            1, 1, self.kernel_size, padding=1, padding_mode="circular")
        self.dx = dx

        weights = torch.tensor(
            [-0.5,  0.0,  0.5], dtype=torch.float32).view(1, 1, self.kernel_size)
        bias = torch.tensor([0.0], dtype=torch.float32)
        self.conv_layer.weight = nn.Parameter(weights)
        self.conv_layer.bias = nn.Parameter(bias)

        for p in self.conv_layer.parameters():
            p.requires_grad = False

    def forward(self, u):
        """
        write the dimension of the forward
        input: 3 dimensional vecter batch_size, variable_size, dx 
        """
        x_len = list(u.shape)[2]

        left_padder = u[:, :, x_len-self.half_padding: x_len]
        right_padder = u[:, :, 0: self.half_padding]
        u_pad = torch.cat([left_padder, u, right_padder],
                          dim=2)
        # 求导
        u_pad_forward = self.conv_layer(u_pad)
        # 对 pad后的区域 cut
        result = (u_pad_forward[:, :, self.half_padding: x_len +
                                self.half_padding]/self.dx)
        return result


fun = Order2_Diff1(dx=0.01)
a = torch.randn(10, 1, 100)
b = fun(a)
print(b.shape)
