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
from dynamic_model.integral_module import Jrk4_step, Jrk4_step2
# , Jrk4_step2
# 4dvar


# def getDM(inverse_time_forward_method, input_time, state):
#     DM = inverse_time_forward_method(input_time, state)
#     return DM

def classical3dvar(observe_ops, state_current, obs_current, R_mat, B_mat):
    """
    A = R + H@B@(H.T) 
    b = (w-H@ub) 
    ua = ub + B@(H.T)@np.linalg.solve(A,b) #solve a linear system
    """
    state_len = len(state_current.shape)
    state_numel = state_current.shape.numel()

    jacobian_h = jacobian(observe_ops, state_current)

    if(state_len == 1):
        A = R_mat + torch.einsum("ab,bc,dc->ad",
                                 jacobian_h, B_mat, jacobian_h)
    if (state_len == 2):
        A = R_mat + torch.einsum("abcd,cdef,ghef->abgh",
                                 jacobian_h, B_mat, jacobian_h)

    if (state_len == 3):
        A = R_mat + torch.einsum("abcdef,defghi,jklghi->abcjkl",
                                 jacobian_h, B_mat, jacobian_h)

    diff = obs_current - observe_ops(state_current)

    # 这个solver只能解矩阵
    A_faltten = A.reshape(state_numel, state_numel)
    diff_flatten = diff.flatten()
    A_solver_flatten = torch.linalg.solve(A_faltten, diff_flatten)
    A_solver = A_solver_flatten.reshape(state_current.shape)

    if(state_len == 1):
        state_current = state_current + \
            torch.einsum("ab,cb,c->a", B_mat, jacobian_h, A_solver)

    if(state_len == 2):
        state_current = state_current + \
            torch.einsum("abcd,efcd,ef->ab", B_mat, jacobian_h, A_solver)

    if(state_len == 3):
        state_current = state_current + \
            torch.einsum("abcdef,ghidef,ghi->abc", B_mat, jacobian_h, A_solver)

    return state_current


def classical4dvar(state_full_pred, state_full_obs, func, time_vector,
                   observe_ops, ind_m, R_inv, R_inv_type, state_background, state_init_pred, B_inv):
    """

    Parameters
    -----------
    state_full_pred: full state of the prediction
    state_full_obs:  full state of the obs field
    DM_method:
    observe_ops: observe function
    ind_m:  index of measurement
    R_inv: inverse of the R matrix
    R_inv_type: integer, 0: only 1 value 1: vector that contains diagonal value 2: full of R_inv matrix


    -----------
    Returns
    -----------

    -----------
    """

    k = ind_m[-1]
    nt = k + 1
    fk = torch.empty(tuple(ind_m.shape) +
                     tuple(state_full_obs.shape[1::]))

    state_len = len(state_full_obs.shape[1::])
    print(f"state_len:{state_len}")
    """
    todo: make it beautiful when i have time
    if the einsum operation looks wield, it is highly likely transpose
    """
    if (state_len == 1):
        mul_op_3vector_type2 = "ba,bc,c->a"
        mul_op_3vector_type0 = "ba,b->a"

        transpose_string = "ab->ba"
        mul_op_2vector = "ab,b->a"

    elif (state_len == 2):
        mul_op_3vector_type2 = "cdab,cdef,ef->ab"
        mul_op_3vector_type0 = "cdab,cd->ab"

        transpose_string = "abcd->cdab"
        mul_op_2vector = "abcd,cd->ab"

    elif (state_len == 3):
        mul_op_3vector_type2 = "defabc,defghi,ghi->abc"
        mul_op_3vector_type0 = "defabc,def->abc"

        transpose_string = "abcdef->defabc"
        mul_op_2vector = "abcdef,def->abc"

    elif (state_len == 4):
        mul_op_3vector_type2 = "efghabcd,efghijkl,ijkl->abcd"
        mul_op_3vector_type0 = "efghabcd,efgh->abcd"

        transpose_string = "abcdefgh->efghabcd"
        mul_op_2vector = "abcdefgh,efgh->abcd"

    # print("checkpoint 0")
    lam = torch.empty((nt,) + tuple(state_full_obs.shape[1::]))

    jacobian_h = jacobian(observe_ops, state_full_pred[k, :])
    # print("checkpoint 0.5")

    if (R_inv_type == 0):
        fk[-1, :] = R_inv*torch.einsum(mul_op_3vector_type0, jacobian_h,
                                       (state_full_obs[-1, :] - state_full_pred[k, :]))

    if(R_inv_type == 2):
        fk[-1, :] = torch.einsum(mul_op_3vector_type2, jacobian_h,
                                 R_inv, (state_full_obs[-1, :] - state_full_pred[k, :]))

    # temp1 = torch.einsum(mul_op_3vector_step1, jacobian_h_transpose, R_inv)
    # fk[-1, :] = torch.einsum(mul_op_3vector_step2, temp1,
    #                          (state_full_obs[k, :] - state_full_pred[k, :]))

    # print("checkpoint 0.6")
    lam[k, :] = fk[-1, :]  # lambda_N = f_N
    # print("checkpoint 1")

    km = len(ind_m) - 2
    for k in range(ind_m[-1], 0, -1):
        predict_ = state_full_pred[k - 1, :]
        true_ = state_full_obs[km, :]

        DM = Jrk4_step(
            func=func, t0=time_vector[k-1], dt=time_vector[k] - time_vector[k-1], y0=predict_)

        # DM = Jrk4_step2(
        #     t0=time_vector[k-1], dt=time_vector[k] - time_vector[k-1], y0=predict_)

        # print("checkpoint 2")
        DM_transpose = torch.einsum(transpose_string, DM)
        lam[k - 1, :] = torch.einsum(mul_op_2vector,
                                     DM_transpose, lam[k, :])

        if ((k - 1) == ind_m[km]):
            # print("checkpoint 3")
            jacobian_h = jacobian(observe_ops, predict_)
            if (R_inv_type == 0):
                fk[km, :] = R_inv*torch.einsum(mul_op_3vector_type0, jacobian_h, (
                    true_ - observe_ops(predict_)))

            if(R_inv_type == 2):
                fk[km, :] = torch.einsum(mul_op_3vector_type2, jacobian_h, R_inv, (
                    true_ - observe_ops(predict_)))

            lam[k - 1, :] = lam[k - 1, :] + fk[km, :]
            km = km - 1
            # print("checkpoint 4")

    if ((state_background is not None) and (state_init_pred is not None) and (B_inv is not None)):

        dJ0 = torch.einsum(mul_op_2vector,
                           B_inv, state_init_pred - state_background) - lam[0, :]
    else:
        dJ0 = -lam[0, :]
    return dJ0


def standard_da_loss(pred_input, true_input, R_inv, R_inv_type, state_background, state_init_pred, B_inv):
    """

    Parameters
    -----------
    pred_input: full state of the prediction
    true_input:  full state of the obs field
    R_inv:  inverse of the R matrix
    R_inv_type: integer, 0: only 1 value 1: vector that contains diagonal value 2: full of R_inv matrix
    state_background: background field
    B_inv: inverse of B matrix
    -----------
    Returns
    -----------
    loss:
    -----------
    """
    diff = pred_input - true_input

    state_len = len(pred_input.shape[1::])

    if (state_len == 1):
        op_string = "na,ab,nb->"
        op_string_background = "a,ab,b->"
    if (state_len == 2):
        op_string = "nab,abcd,ncd->"
        op_string_background = "ab,abcd,cd->"
    if (state_len == 3):
        op_string = "nabc,abcdef,ndef->"
        op_string_background = "abc,abcdef,def->"

    if(R_inv_type == 2):
        if((state_background is not None) and (state_init_pred is not None) and (B_inv is not None)):
            diff_back = state_init_pred - state_background
            loss = torch.einsum(op_string, diff, R_inv, diff) / 2.0 + \
                torch.einsum(op_string_background,
                             diff_back, B_inv, diff_back) / 2.0
        else:
            loss = torch.einsum(op_string, diff, R_inv, diff) / 2.0

    if (R_inv_type == 0):
        loss = R_inv*torch.sum(diff**2) / 2.0

    return loss


def main():
    pass


if __name__ == "__main__":
    main()
