#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen
"""


class RunningAverageMeter(object):
    """Computes and stores the average and current value
        for smoothing of the time series
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.count = self.count + 1
        self.sum = self.sum + self.val
