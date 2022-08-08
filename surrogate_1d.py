#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:10:04 2021

@author: yaoyichen
"""
# %%
import os
import torch
import torch.nn as nn
import time
import numpy as np
import argparse
import logging
from neural_model.differ_module import Order2_Diff1, Order2_Diff2
from torch.utils.tensorboard import SummaryWriter

from tools.statistical_helper import RunningAverageMeter
from tools.model_helper import ModelUtils
from tools.plot_helper import generate_line_movement_gif
from tools.file_helper import FileUtils
from neural_model.surrogate_module import FC_Model_W4, FC_Model_N4, CNN_Model_W4, CNN_Model_N4, PDE_Model_Test1
import torch.optim as optim


class Args:
    method = "rk4"  # "rk4"
    data_size = 200
    batch_time = 10
    batch_size = 100
    niters = 2000

    viz = True
    gpu = 0
    adjoint = False

    main_folder = "surrogate_1d"   # 问题定义
    sub_folder = "cnn"             # 主方案定义
    prefix = "cnn_test1"
    # prefix = "cnn_fun"
    surrogate_mode = "cnn_n4"

    load_model = False
    save_model = True
    source_checkpoint_file_name = "test1.pth.tar"
    target_checkpoint_file_name = "test2.pth.tar"
    test_freq = 1
    save_interval = 100
    plot_interval = 100


#
parser = argparse.ArgumentParser('Surrogate 1d')
parser.add_argument('--method', type=str,
                    choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--data_size', type=int, default=300)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)

parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--adjoint', type=eval,
                    default=False, choices=[True, False])

parser.add_argument('--main_folder', type=str, default='rk4')
parser.add_argument('--sub_folder', type=str, default='rk4')
parser.add_argument('--prefix', type=str, default='rk4')

parser.add_argument('--surrogate_mode', type=str,
                    choices=['fc', 'cnn_n4', 'cnn_w4', 'pde'], default='pde')

parser.add_argument('--load_model', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--save_model', type=eval,
                    default=True, choices=[True, False])

parser.add_argument('--source_checkpoint_file_name', default="test1.pth.tar")
parser.add_argument('--target_checkpoint_file_name', default="test1.pth.tar")


parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--plot_interval', type=int, default=100)


args = Args()
# args = parser.parse_args()

# parser.add_argument('--test_freq', type=int, default=20)


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


FileUtils.makedir(os.path.join(
    "runs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(
    "logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(
    "results", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(
    "checkpoints", args.main_folder, args.sub_folder))


filehandler = logging.FileHandler(os.path.join(
    "logs", args.main_folder, args.sub_folder, args.prefix + ".txt"))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)

random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)

time_str = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
writer = SummaryWriter(os.path.join(
    "runs", args.main_folder, args.sub_folder, args.prefix + time_str))

# for debug purpose
torch.autograd.set_detect_anomaly(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device == "cuda"):
    torch.backends.cudnn.benchmark = True


def construct_initial_state():
    """
    construct the initial problem definition
    """
    dx = 0.01
    x = torch.arange(0, 100 * dx, dx, dtype=torch.float32)

    grid_info = (dx, x)

    u = 1.0 + torch.sin(2 * np.pi * x).to(torch.float32)

    u0_true = 1.0 \
        + 0.2 * torch.sin(2*np.pi * x) \
        + 0.1*torch.cos(6*np.pi*x + 1.0/3.0) \
        + 0.1*torch.sin(10*np.pi*x + 5.0/9.0) \
        + 0.01 * torch.randn(list(x.shape)[0])

    u = u.unsqueeze(0).unsqueeze(0)
    u.requires_grad = True
    u0_true = u0_true.unsqueeze(0).unsqueeze(0)
    return grid_info, u, u0_true


(dx, x), u, u0_true = construct_initial_state()
x, u, u0_true = x.to(device), u.to(device), u0_true.to(device)

print("u shape:", u.shape, "u0 shape::", u0_true.shape)


class PDE_Model(nn.Module):
    def __init__(self, dx, zero_diff, first_diff, second_diff):
        super(PDE_Model, self).__init__()
        self.term1 = Order2_Diff1(dx, half_padding=1)
        self.term2 = Order2_Diff2(dx, half_padding=1)
        self.zero_diff = zero_diff
        self.first_diff = first_diff
        self.second_diff = second_diff

    def forward(self, t, u, zero_diff=(0, 1), first_diff=(0, 1), second_diff=(0, 1)):

        if(False):
            u_clone = u.clone()
            for term, value in enumerate(self.zero_diff):
                result = result + value*torch.pow(u_clone, term)*u

            for term, value in enumerate(self.first_diff):
                result = result + value*torch.pow(u_clone, term)*self.term1(u)

            for term, value in enumerate(self.second_diff):
                result = result + value*torch.pow(u_clone, term)*self.term2(u)
        result = (- (u + 0.5) * self.term1.forward(u) +
                  0.0001 * self.term2.forward(u))
        return result


# pde model
pde_model = PDE_Model(
    dx, zero_diff=(), first_diff=(-0.5, -1.0), second_diff=(1e-4,)).to(device)

t_full = torch.linspace(0., 0.9, args.data_size).to(device)
y_true = odeint(pde_model, u0_true, t_full,
                method=args.method)
y_true = y_true.to(device)


if(False):
    generate_line_movement_gif(x=x.cpu().detach().numpy(),
                               y=y_true.squeeze().cpu().detach().numpy(), file_name=os.path.join("results", args.main_folder, args.sub_folder, args.prefix + "true.gif"),
                               fps=20, xlim=(0, 1), ylim=(0.0, 2.0))

# %%


def get_batch(input_, args):
    """
    y_true:     [1,1,100]
    batch_y0:   [Batch,1,100]
    batch_y:    [Time, Batch,1,100]
    """
    s = torch.from_numpy(np.random.choice(np.arange(
        args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = input_[s]  # (M, D)
    batch_t = t_full[:args.batch_time]  # (T)
    batch_y = torch.stack([input_[s + i]
                           for i in range(args.batch_time)], dim=0)  # (T, M, D)

    batch_y0 = batch_y0.squeeze(1)
    batch_y = batch_y.squeeze(2)
    return batch_y0, batch_t, batch_y


print("y_true shape:", y_true.shape)
batch_y0, batch_t, batch_y = get_batch(y_true, args)
print(batch_y0.shape, batch_t.shape, batch_y.shape)


def plot_result(x, u0_true, y_true, u, y_pred, file_name):  # %%
    if (args.viz):
        import matplotlib.pyplot as plt
        print("=> ploting_result")
        with torch.no_grad():
            plt.figure(0)
            plt.clf()
            plt.title("")
            plt.plot(x.cpu().detach().numpy(), u0_true.cpu().squeeze(
            ).detach().numpy(), '-k', label="true_start")
            plt.plot(x.cpu().detach().numpy(), y_true[-1, 0, :].cpu().squeeze().detach(
            ).numpy(), label="true_end")

            plt.plot(x.cpu().detach().numpy(), u.cpu().squeeze(
            ).detach().numpy(), 'tab:gray', label="pred_start")
            plt.plot(x.cpu().detach().numpy(), y_pred[-1, 0, :].cpu().squeeze().detach(
            ).numpy(), label="pred_end")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("results", args.main_folder,
                                     args.sub_folder, file_name))


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


if __name__ == '__main__':
    # CNN_Func
    if(args.surrogate_mode == "fc"):
        surrogate_model = FC_Model_W4(
            n_features=100, hidden_dims=(100, 100), p=0.2).to(device)

        # surrogate_model = FC_Model_N4(
        #     n_features=100, p=0.2).to(device)

    if(args.surrogate_mode == "cnn_w4"):
        surrogate_model = CNN_Model_W4(
            input_channel=1, hidden_channels=(10, 10, )).to(device)

    if(args.surrogate_mode == "cnn_n4"):
        surrogate_model = CNN_Model_N4(
            input_channel=1).to(device)

    if (args.surrogate_mode == "pde"):
        surrogate_model = PDE_Model_Test1(dx=0.01).to(device)

        # surrogate_model = CNN_Func(input_channel=1)
        # input_channel=1, hidden_channels=None, p=0, augment_factor=100).to(device)

    # print(surrogate_model.state_dict())
    ModelUtils.print_model_layer(surrogate_model)
    parameter_reuslt = ModelUtils.get_parameter_number(surrogate_model)
    logger.info(surrogate_model)
    logger.info(parameter_reuslt)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=1.0e-4)

    if(args.load_model):
        surrogate_model, optimizer = load_checkpoint(torch.load(os.path.join("checkpoints", args.main_folder,
                                                                             args.sub_folder, args.source_checkpoint_file_name)),
                                                     surrogate_model,
                                                     optimizer)
    total_time = 0
    total_time0 = 0
    total_time1 = 0
    total_time2 = 0

    for iteration in range(1, args.niters + 1):

        start_time0 = time.time()
        batch_y0, batch_t, batch_y = get_batch(y_true, args)
        batch_y0, batch_t, batch_y = batch_y0.to(
            device), batch_t.to(device), batch_y.to(device)
        total_time0 = total_time0 + time.time() - start_time0

        start_time = time.time()
        y_pred = odeint(surrogate_model, batch_y0,
                        batch_t, rtol=1e50, atol=1e50)
        total_time1 = total_time1 + time.time() - start_time

        start_time = time.time()
        optimizer.zero_grad()
        loss = criterion(y_pred[:, :], batch_y[:, :])
        loss.backward()
        optimizer.step()
        total_time2 = total_time2 + time.time() - start_time

        total_time = total_time + (time.time() - start_time0)

        torch.cuda.empty_cache()

        if (iteration + 1) % args.test_freq == 0:
            average_time = total_time / args.test_freq
            average_time0 = total_time0 / args.test_freq
            average_time1 = total_time1 / args.test_freq
            average_time2 = total_time2 / args.test_freq
            total_time = 0
            total_time0 = 0
            total_time1 = 0
            total_time2 = 0
            with torch.no_grad():
                surrogate_model.eval()
                y_pred_total = odeint(
                    surrogate_model, u0_true, t_full)
                test_loss = criterion(y_pred_total, y_true)
                logger.info('Iter {:04d} | Time :{:.3f} | Train Loss {:.6f} | Total Loss {:.6f}'.format(
                    iteration + 1, average_time, loss.item(), test_loss.item()))
                # logger.info('Iter: {:04d}  |Time0 {:.6f} | Time1 :{:.6f} | Time2: {:.6f}'.format(
                #     iteration + 1, average_time0, average_time1, average_time2))
                surrogate_model.train()

        if (((iteration+1) % args.plot_interval == 0)):
            file_name = args.prefix + "result_conv" + \
                str(iteration+1).zfill(4) + ".png"
            plot_result(x, u0_true, y_true, u0_true, y_pred_total, file_name)

            generate_line_movement_gif(x=x.cpu().detach().numpy(),
                                       y=y_pred_total.squeeze().cpu().detach().numpy(),
                                       file_name=os.path.join(
                                           "results", args.main_folder, args.sub_folder, args.prefix + str(iteration+1) + ".gif"),
                                       fps=20, xlim=(0, 1), ylim=(0.0, 2.0))

        if (((iteration + 1) % args.save_interval == 0) and args.save_model):
            checkpoint = {"state_dict": surrogate_model.state_dict(),
                          "optimizer": optimizer.state_dict()}
            ModelUtils.save_checkpoint(
                checkpoint, filename=os.path.join("checkpoints", args.main_folder,
                                                  args.sub_folder, args.target_checkpoint_file_name))
