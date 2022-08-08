import torch
import torchcde

# Create some data
batch, length, input_channels = 1, 10, 2
hidden_channels = 3
t = torch.linspace(0, 1, length)
t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
x_ = torch.rand(batch, length, input_channels - 1)
x = torch.cat([t_, x_], dim=2)  # include time as a channel

# Interpolate it
coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
X = torchcde.CubicSpline(coeffs)

# Create the Neural CDE system


class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.linear = torch.nn.Linear(hidden_channels,
                                      hidden_channels * input_channels)

    def forward(self, t, z):
        return self.linear(z).view(batch, hidden_channels, input_channels)


func = F()
z0 = torch.rand(batch, hidden_channels)

# Integrate it
result = torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval)
print(z0.shape, result.shape, t.shape)
