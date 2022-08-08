#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn
from neural_model.differ_module import Order2_Diff1, Order2_Diff2
import torch.nn.functional as F


class FC_Model_W4(nn.Module):
    def __init__(self, n_features, hidden_dims, p=0):
        super(FC_Model_W4, self).__init__()
        dims = (n_features,) + hidden_dims
        self.n_hidden = len(hidden_dims)

        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dims[i], dims[i + 1]),
                           nn.BatchNorm1d(1),
                           nn.Tanh(),
                           )
             for i in range(self.n_hidden)
             ]
        )
        self.clf = nn.Sequential(nn.Linear(dims[-1], n_features),
                                 nn.Conv1d(1, 1, kernel_size=1, padding=0))

        for m in self.hidden_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.0001)
                nn.init.constant_(m.bias, val=0)

        for m in self.clf.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.0001)
                nn.init.constant_(m.bias, val=0)

        for m in self.clf.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=100.0, std=30)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        for model in self.hidden_layers:
            state = model(state)
        return self.clf(state)


class FC_Model_N4(nn.Module):
    def __init__(self, n_features, p=0):
        super(FC_Model_N4, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(1),
            nn.Tanh(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(1),
            nn.Tanh(),

            nn.Linear(100, 100),
            nn.Conv1d(1, 1, kernel_size=1, padding=0)
        )

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=1.0e-3, std=1e-3)
                nn.init.constant_(m.bias, val=0)

        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=100.0, std=30)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        state = self.model(state)
        return state

# class Surrogate_Model_FC2(nn.Module):
#     def __init__(self, n_features, hidden_dims, p=0):
#         super(Surrogate_Model_FC2, self).__init__()
#         dims = (n_features,) + hidden_dims
#         self.n_hidden = len(hidden_dims)

#         self.hidden_layers = nn.ModuleList(
#             [nn.Sequential(nn.Linear(dims[i], dims[i+1]),
#                            nn.LeakyReLU(0.2),
#                            nn.Dropout(p)
#                            )
#              for i in range(self.n_hidden)
#              ]
#         )
#         self.clf = nn.Linear(dims[-1], n_features)

#         for m in self.hidden_layers.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.0001)
#                 nn.init.constant_(m.bias, val=0)

#         for m in self.clf.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.0001)
#                 nn.init.constant_(m.bias, val=0)

#     def forward(self, u):
#         u_add = u.clone()
#         for model in self.hidden_layers:
#             u_add = model(u_add)
#         u_add = self.clf(u_add)
#         return u + u_add


class CNN_Model_W4(nn.Module):
    def __init__(self, input_channel, hidden_channels):
        super(CNN_Model_W4, self).__init__()
        dims = (input_channel,) + hidden_channels
        self.n_hidden = len(hidden_channels)
        self.kernel_size = 3

        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(dims[i], dims[i + 1], kernel_size=self.kernel_size, padding=self.kernel_size // 2, padding_mode="circular"),
                           nn.BatchNorm1d(dims[i + 1]),
                           nn.Tanh()
                           )
             for i in range(self.n_hidden)
             ]
        )
        self.clf_conv = nn.Conv1d(dims[-1], input_channel,
                                  kernel_size=self.kernel_size,
                                  padding=self.kernel_size//2,
                                  padding_mode="circular")

        self.clf_sacle = nn.Conv1d(
            input_channel, input_channel, kernel_size=1, padding=0)

        for m in self.hidden_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.clf_conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.clf_sacle.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=100.0, std=1.0)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, state):
        for model in self.hidden_layers:
            state = model(state)
        state = self.clf_conv(state)
        return self.clf_sacle(state)


class FC_Func(nn.Module):

    def __init__(self):
        super(FC_Func, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-6)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        return self.net(state)


class CNN_Model_N4(nn.Module):
    def __init__(self, input_channel):
        super(CNN_Model_N4, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channel, 10, kernel_size=3, padding=1,
                      padding_mode="circular"),
            nn.Tanh(),
            nn.BatchNorm1d(10),

            nn.Conv1d(10, 10, kernel_size=3, padding=1,
                      padding_mode="circular"),
            nn.Tanh(),
            nn.BatchNorm1d(10),

            nn.Conv1d(10, 10, kernel_size=3, padding=1,
                      padding_mode="circular"),
            nn.Tanh(),
            nn.BatchNorm1d(10),


            nn.Conv1d(10, 10, kernel_size=3, padding=1,
                      padding_mode="circular"),
            nn.Tanh(),
            nn.BatchNorm1d(10),

            nn.Conv1d(10, input_channel, kernel_size=1, padding=0,
                      padding_mode="circular"),
        )

        self.net.add_module("scale",   nn.Conv1d(
            1, 1, kernel_size=1, padding=0))

        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

        for name, module in self.net.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=100, std=1.0)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        result = self.net(state)
        return result


class Constant_Module(nn.Module):
    def __init__(self, value):
        super(Constant_Module, self).__init__()
        self.value = torch.tensor(value)

    def forward(self, state):
        return state*self.value


class PDE_Model_Test1(nn.Module):
    def __init__(self, dx):

        super(PDE_Model_Test1, self).__init__()
        self.dx = dx
        self.term1 = Order2_Diff1(dx, half_padding=1)
        self.term2 = Order2_Diff2(dx, half_padding=1)
        self.c = Constant_Module(100)

        self.net = nn.Sequential(
            nn.Conv1d(4, 10, kernel_size=1, padding=0,
                      padding_mode="circular"),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            nn.Conv1d(10, 10, kernel_size=1, padding=0,
                      padding_mode="circular"),
            nn.BatchNorm1d(10),
            nn.Tanh(),

            nn.Conv1d(10, 1, kernel_size=1, padding=0,
                      padding_mode="circular"),
        )

        self.net.add_module("scale",   nn.Conv1d(
            1, 1, kernel_size=1, padding=0, bias=True))

        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

        nn.init.normal_(self.net.scale.weight, mean=100, std=1.0)
        nn.init.constant_(self.net.scale.bias, val=0)

    def forward(self, t, state):
        original_shape = state.shape
        term1 = self.term1(state)
        term2 = state * self.term1(state)
        term3 = self.term2(state)
        features = torch.cat(
            [state, term1, term2, term3], dim=1)

        result = self.net(features)
        return result


def test_net():
    import sys
    sys.path.insert(1, '../tools')
    import model_helper

    x = torch.randn([5, 1, 100])
    # print(x.mean())

    # model = FC_Model_W4(n_features=100, hidden_dimscd =(100, 100))
    # model = FC_Model_N4(n_features=100, p=0.2)
    # model = CNN_Model_W4(input_channel=1, hidden_channels=(10, 10, 10))
    model = CNN_Model_N4(input_channel=1)
    model = PDE_Model_Test1(dx=0.01)

    model_helper.ModelUtils.print_model_layer(model)
    parameter_reuslt = model_helper.ModelUtils.get_parameter_number(
        model)
    print(model)
    print(parameter_reuslt)

    # model = CNN_Model(input_channel=3, hidden_channels=(5, 5, 5, 5), p=0.2)
    print(model)

    out = model(0, state=x)
    # print(out.mean())
    print(out.shape)


if __name__ == "__main__":
    test_net()
