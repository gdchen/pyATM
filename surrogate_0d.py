
# %%
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Args:
    method = "rk4"  # "rk4"
    data_size = 200
    batch_time = 5
    batch_size = 100
    niters = 4000
    test_freq = 20
    viz = True
    gpu = 0
    adjoint = False


args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

device = "cpu"

true_y0 = torch.tensor([1.0, 1.0, 1.0]).to(device)
t = torch.linspace(0., 2.0, args.data_size).to(device)
dt = t[1] - t[0]


def plot_Lorenz63(time, state, time_obs, obs, fig_index):
    """
    plot the basic figure of the Lorenz63 model
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(fig_index, figsize=(6, 4))
    plt.plot(time, state[:, 0], c="tab:blue")
    plt.plot(time, state[:, 1], c="tab:orange")
    plt.plot(time, state[:, 2], c="tab:green")
    # plt.plot(time_obs, obs[:,0], 'x', c="tab:blue")
    # plt.plot(time_obs, obs[1, :], 'x', c="tab:orange")
    # plt.plot(time_obs, obs[2, :], 'x', c="tab:green")
    plt.xlabel("time")
    plt.ylabel("state value")


class Lorenz63(nn.Module):  # Lorenz 96 model
    def __init__(self, *args):
        super(Lorenz63, self).__init__()
        print(args)
        self.sigma = args[0]
        self.beta = args[1]
        self.rho = args[2]

    def forward(self, t, state):
        # Unpack the state vector
        x, y, z = state[0], state[1], state[2]
        f = torch.zeros(state.shape)  # Derivatives
        f[0] = self.sigma * (y - x)
        f[1] = x * (self.rho - z) - y
        f[2] = x * y - self.beta * z
        return f


sigma = 10.0
beta = 8.0/3.0
rho = 28.0


with torch.no_grad():
    true_y = odeint(Lorenz63(sigma, beta, rho), true_y0, t, method='rk4')
    print(true_y.shape)


print("##" * 30)
plot_Lorenz63(t, true_y, t, true_y, 0)


# %%
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(
        args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i]
                           for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


batch_y0, batch_t, batch_y = get_batch()
print(batch_y0.shape, batch_t.shape, batch_y.shape)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.BatchNorm1d(num_features=50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-6)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):

        return self.net(state) / dt


class RightFunc(nn.Module):

    def __init__(self):
        super(RightFunc, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(9, 50),
            nn.Tanh(),
            # nn.BatchNorm1d(50),
            nn.Linear(50, 9),
            nn.Tanh(),
            # nn.BatchNorm1d(9),
            nn.Linear(9, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-7)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        x, y, z = state[::, 0], state[::, 1], state[::, 2]
        term1 = x - y
        term2 = y - z
        term3 = z - x
        term4 = x * y
        term5 = y * z
        term6 = x * z
        feature = torch.stack(
            [x, y, z, term1, term2, term3, term4, term5, term6], dim=1)
        return self.net(feature) / dt


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    criterion = nn.MSELoss()

    optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()

        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = criterion(pred_y[-1, :], batch_y[-1, :])
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0.unsqueeze(0), t)
                loss = criterion(pred_y, true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(
                    itr, loss.item()))
                ii += 1

        end = time.time()


# %%
