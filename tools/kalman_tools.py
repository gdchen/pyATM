#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:50:58 2021

@author: yaoyichen
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import torch
from torch.autograd.functional import jacobian
from dynamic_model.integral_module import Jrk4_step


def KF_cal_Bmatrix(B_matrix, pde_model, t0, dt, state_current):
    state_len = len(state_current.shape)
    DM = Jrk4_step(
        func=pde_model, t0=t0, dt=dt, y0=state_current)

    if(state_len == 1):
        B_matrix = torch.einsum("ab,bc,dc->ad", DM, B_matrix, DM)
    if(state_len == 2):
        B_matrix = torch.einsum("abcd,cdef,ghef->abgh", DM, B_matrix, DM)
    if(state_len == 3):
        B_matrix = torch.einsum(
            "abcdef,defghi,jklghi->abcjkl", DM, B_matrix, DM)

    return B_matrix


def KF_update_state_corvariance_ensemble(state_current_batch):

    state_current_batch_mean = torch.mean(state_current_batch, dim=0)

    # 更新B矩阵
    state_current_batch_fluctuation = state_current_batch - state_current_batch_mean

    P_matrix = KF_cal_Bmatrix_ensemble(state_current_batch_fluctuation)

    return state_current_batch_mean, P_matrix


def KF_cal_Bmatrix_ensemble(state_current_batch_fluctuation):
    state_len = len(state_current_batch_fluctuation.shape[1::])

    if(state_len == 1):
        B_matrix = torch.mean(torch.einsum("na,nb->nab", state_current_batch_fluctuation,
                                           state_current_batch_fluctuation), dim=0)
    if(state_len == 2):
        B_matrix = torch.mean(torch.einsum("nab,ncd->nabcd", state_current_batch_fluctuation,
                                           state_current_batch_fluctuation), dim=0)
    if (state_len == 3):
        print(state_current_batch_fluctuation.shape)
        # tt = torch.mean(torch.einsum("nabcd,nefgh->nabcdefgh", state_current_batch_fluctuation,
        #                              state_current_batch_fluctuation))
        # print(tt.shape)
        B_matrix = torch.mean(torch.einsum("nabc,ndef->nabcdef", state_current_batch_fluctuation,
                                           state_current_batch_fluctuation), dim=0)
    return B_matrix


def KF_cal_Kmatrix(state_current, jacobian_h, B_matrix, R_mat):
    """
    # ###  step 2  kilman gain ###
    # compute Jacobian of observation operator at ub
    # Dh = JObsOp(ub),  D = Dh @ B @ Dh.T
    # K = B @ Dh.T @ np.linalg.inv(D)
    """
    state_len = len(state_current.shape)

    #
    state_shape = state_current.shape
    state_numel = state_shape.numel()

    if(state_len == 1):
        D_matrix = torch.einsum(
            "ab,bc,dc->ad", jacobian_h, B_matrix, jacobian_h) + R_mat
    if(state_len == 2):
        D_matrix = torch.einsum(
            "abcd,cdef,ghef->abgh", jacobian_h, B_matrix, jacobian_h) + R_mat
    if(state_len == 3):
        D_matrix = torch.einsum(
            "abcdef,defghi,jklghi->abcjkl", jacobian_h, B_matrix, jacobian_h) + R_mat

    # D_matrix_inv = torch.linalg.inv(D_matrix.reshape(100, 100))
    # D_matrix_inv_reshape = D_matrix_inv.reshape([1, 100, 1, 100])

    D_matrix_inv = torch.linalg.inv(D_matrix.reshape(state_numel, state_numel))
    D_matrix_inv_reshape = D_matrix_inv.reshape(state_shape + state_shape)

    if(state_len == 1):
        K_matrix = torch.einsum(
            "ab,cb,cd->ad", B_matrix, jacobian_h, D_matrix_inv_reshape)
    if(state_len == 2):
        K_matrix = torch.einsum(
            "abcd,efcd,efgh->abgh", B_matrix, jacobian_h, D_matrix_inv_reshape)
    if(state_len == 3):
        K_matrix = torch.einsum(
            "abcdef,ghidef,ghijkl->abcjkl", B_matrix, jacobian_h, D_matrix_inv_reshape)

    return K_matrix


def KF_update_state(state_current, K_matrix, diff):
    """
    """
    state_len = len(state_current.shape)

    if(state_len == 1):
        state_current = state_current + \
            torch.einsum("ab,b->a", K_matrix, diff)
    if(state_len == 2):
        state_current = state_current + \
            torch.einsum("abcd,cd->ab", K_matrix, diff)
    if(state_len == 3):
        state_current = state_current + \
            torch.einsum("abcdef,def->abc", K_matrix, diff)

    return state_current


def KF_update_state_batch(state_current_batch, K_matrix, diff_batch):
    """
    """
    state_len = len(state_current_batch.shape[1::])

    if(state_len == 1):
        state_current_batch = state_current_batch + \
            torch.einsum("ab,nb->na", K_matrix, diff_batch)
    if(state_len == 2):
        state_current_batch = state_current_batch + \
            torch.einsum("abcd,ncd->nab", K_matrix, diff_batch)
    if(state_len == 3):
        state_current_batch = state_current_batch + \
            torch.einsum("abcdef,ndef->nabc", K_matrix, diff_batch)

    return state_current_batch


def KF_update_corvariance(K_matrix, jacobian_h, B_matrix):

    state_len = len(K_matrix.shape)//2
    state_shape = K_matrix.shape[0:state_len]
    state_numel = state_shape.numel()

    if(state_len == 1):
        P_factor = torch.einsum(
            "ab,bc->ac", K_matrix, jacobian_h)

    if(state_len == 2):
        P_factor = torch.einsum(
            "abcd,cdef->abef", K_matrix, jacobian_h)

    if(state_len == 3):
        P_factor = torch.einsum(
            "abcdef,defghi->abcghi", K_matrix, jacobian_h)

    P_identity = torch.eye(state_numel).reshape(state_shape + state_shape)

    if(state_len == 1):
        P_matrix = torch.einsum(
            "ab,bc->ac", (P_identity - P_factor), B_matrix)

    if(state_len == 2):
        P_matrix = torch.einsum(
            "abcd,cdef->abef", (P_identity - P_factor), B_matrix)

    if(state_len == 3):
        P_matrix = torch.einsum(
            "abcdef,defghi->abcghi", (P_identity - P_factor), B_matrix)

    return P_matrix


def main():
    pass


if __name__ == "__main__":
    main()
