#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:50:58 2021

@author: yaoyichen
"""
import torch


def standard_da_loss(pred_input, , R_inv):

    diff = pred_input - true_input
    state_len = len(pred_input.shape[1::])

    transpose_string = "nab->nba"
    diff_transpose = torch.einsum(transpose_string, diff)
    test_string = "nab,abcd->ncd"
    diff_test = torch.einsum(test_string, diff, R_inv)
    loss = torch.einsum("nab,nba->", diff_test, diff_transpose) / 2.0

    return loss


def main():
    pass


if __name__ == "__main__":
    main()
