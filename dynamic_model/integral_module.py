#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, vjp
# from dynamic_model.Lorenz63 import construct_lorenz63_initial_state, Lorenz63, plot_Lorenz63_onevector

# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


class Euler_Method(nn.Module):
    def __init__(self, model):
        super(Euler_Method, self).__init__()
        self.model = model

    def forward(self, u, dt):
        return u + dt * self.model(u)


class RK4_Method(nn.Module):
    def __init__(self, model, dt):
        super(RK4_Method, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, u):
        k1 = self.model(u)
        u1 = u + self.dt * k1

        k2 = self.model(u1)
        u2 = u + 0.5*self.dt * k2

        k3 = self.model(u2)
        u3 = u + 0.5*self.dt * k3

        k4 = self.model(u3)

        k = (k1 + 2*k2 + 2*k3 + k4)/(6.0)
        return u + self.dt * k


def rk4_step(func, t0, dt, y0):
    half_dt = dt * 0.5
    device = y0.device
    
    k1 = func(t0, y0).to(device)
    if(k1.shape != y0.shape):
        k1 = torch.reshape(k1, y0.shape)
    
    # print(f"k1.device:{k1.device},y0.device:{y0.device}")
    # print(f"k1 type:{k1.dtype}")
    k2 = func(t0 + half_dt, y0 + half_dt * k1).to(device)
    # print(f"k2 type:{k2.dtype}, half_dt type:{type(half_dt)}")
    if(k2.shape != y0.shape):
        k2 = torch.reshape(k2, y0.shape)
    
    # print(f"k2.device:{k2.device}")
    k3 = func(t0 + half_dt, y0 + half_dt * k2).to(device)
    
    if(k3.shape != y0.shape):
        k3 = torch.reshape(k3, y0.shape)
        
        
    k4 = func(t0 + dt, y0 + dt * k3).to(device)
    
    if(k4.shape != y0.shape):
        k4 = torch.reshape(k4, y0.shape)
        
        
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk2_step(func, t0, dt, y0):
    device = y0.device
    k1 = func(t0, y0).to(device)
    if(k1.shape != y0.shape):
        k1 = torch.reshape(k1, y0.shape)
        
        
    k2 = func(t0 + dt, y0 + dt * k1).to(device)
    
    if(k2.shape != y0.shape):
        k2 = torch.reshape(k2, y0.shape)
        
        
    return (k1 + k2) * dt * 0.5

def rk1_step(func, t0, dt, y0):
    device = y0.device
    k1 = func(t0, y0).to(device)
    if(k1.shape != y0.shape):
        k1 = torch.reshape(k1, y0.shape)
        
        
    return k1*dt


def rk4_step_partial(func, t0, dt, y0, c0, device="cpu"):
    """
    partial of the state is known
    c0 is the condition 
    """
    half_dt = dt * 0.5
    k1 = func(t0, y0, c0).to(device)
    # print(f"k1.device:{k1.device},y0.device:{y0.device}")
    k2 = func(t0 + half_dt, y0 + half_dt * k1, c0).to(device)
    # print(f"k2.device:{k2.device}")
    k3 = func(t0 + half_dt, y0 + half_dt * k2, c0).to(device)
    k4 = func(t0 + dt, y0 + dt * k3, c0).to(device)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth



def euler_step(func, t0, dt, y0, device="cpu"):
    k1 = func(t0, y0).to(device)
    return (k1) * dt * _one_sixth


def rk4_step_temp(t0, dt, y0, device="cpu"):
    half_dt = dt * 0.5
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
    func = Lorenz63(sigma, beta, rho).to(device)

    k1 = func(t0, y0).to(device)
    k2 = func(t0 + half_dt, y0 + half_dt * k1).to(device)
    k3 = func(t0 + half_dt, y0 + half_dt * k2).to(device)
    k4 = func(t0 + dt, y0 + dt * k3).to(device)
    return y0 + (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def Jrk4_step2(t0, dt, y0):
    _0, _1, DM = jacobian(rk4_step_temp, (t0, dt, y0))
    return DM


def Jrk4_step(func, t0, dt, y0):
    state = y0
    state_len = len(state.shape)
    if (state_len == 1):
        op_string = "ab,bc->ac"
    elif (state_len == 2):
        op_string = "abcd,cdef->abef"
    elif (state_len == 3):
        op_string = "abcdef,defghi->abcghi"
    elif (state_len == 4):
        op_string = "abcdefgh,efghijkl->abcdijkl"

    k1 = func(t0, state)
    k2 = func(t0+0.5*dt, state + 0.5*dt * k1)
    k3 = func(t0+0.5*dt, state + 0.5*dt * k2)

    jacobian_identity = torch.eye(
        state.shape.numel()).reshape(state.shape + state.shape)

    _, dk1 = jacobian(func, (t0, state))

    _, dk2 = jacobian(func, (t0 + 0.5 * dt,
                             state + 0.5 * k1 * dt))
    dk2 = torch.einsum(
        op_string, dk2, (jacobian_identity + dk1 * dt / 2))

    _, dk3 = jacobian(func, (t0 + 0.5 * dt,
                             state + 0.5 * k2 * dt))
    dk3 = torch.einsum(
        op_string, dk3, (jacobian_identity + dk2 * dt / 2))
    _, dk4 = jacobian(func, (t0 + dt, state + k3 * dt))

    dk4 = torch.einsum(
        op_string, dk4, (jacobian_identity + dk3 * dt))

    DM = jacobian_identity + _one_sixth * dt * \
        (dk1 + 2 * dk2 + 2 * dk3 + dk4)

    return DM


def da_odeint(func, init_state, time_vector, method, device="cpu"):
    """
    ignore method 
    """
    solver = rk4_step

    full_state = torch.empty(size=tuple(
        time_vector.shape) + tuple(init_state.shape)).to(device)
    full_state[0, :] = init_state

    # print(init_state.device, full_state.device)

    state = init_state
    for index, time in enumerate(time_vector[1::]):
        dt = time_vector[index + 1] - time_vector[index]
        state = state + solver(func, time, dt,  state)
        full_state[index + 1, :] = state
    final_state = full_state[-1, :]
    return final_state, full_state


def da_odeint_boundary(func, init_state, time_vector, method, device="cpu"):
    """
    ignore method 
    """
    solver = rk4_step

    full_state = torch.empty(size=tuple(
        time_vector.shape) + tuple(init_state.shape)).to(device)
    full_state[0, :] = init_state

    state = init_state
    for index, time in enumerate(time_vector[1::]):
        dt = time_vector[index + 1] - time_vector[index]
        state = state + solver(func, time, dt, state, device=device)

        divergence = func.calcualte_divergence(state)
        div = torch.sum(torch.abs(divergence))
        print(f"divergence:{div}")

        full_state[index + 1, :] = state

    final_state = full_state[-1, :]
    return final_state, full_state


class RK4Method(nn.Module):
    def __init__(self, model, dt):
        super(RK4Method, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, t, state):
        k1 = self.model(t, state)
        k2 = self.model(t+0.5*self.dt, state + 0.5*self.dt * k1)
        k3 = self.model(t+0.5*self.dt, state + 0.5*self.dt * k2)
        k4 = self.model(t+self.dt, state + 1.0*self.dt * k3)
        k = _one_sixth*(k1 + 2*k2 + 2*k3 + k4)
        return state + self.dt * k


class JRK4Method(nn.Module):
    def __init__(self, model, dt):
        super(JRK4Method, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, t, state):
        state_len = len(state.shape)
        if (state_len == 1):
            op_string = "ab,bc->ac"
        elif (state_len == 2):
            op_string = "abcd,cdef->abef"
        elif (state_len == 3):
            op_string = "abcdef,defghi->abcghi"
        elif (state_len == 4):
            op_string = "abcdefgh,efghijkl->abcdijkl"

        k1 = self.model(t, state)
        k2 = self.model(t+0.5*self.dt, state + 0.5*self.dt * k1)
        k3 = self.model(t+0.5*self.dt, state + 0.5*self.dt * k2)

        jacobian_identity = torch.eye(
            state.shape.numel()).reshape(state.shape + state.shape)

        _, dk1 = jacobian(self.model, (t, state))

        _, dk2 = jacobian(self.model, (t + 0.5 * self.dt,
                                       state + 0.5 * k1 * self.dt))
        dk2 = torch.einsum(
            op_string, dk2, (jacobian_identity + dk1 * self.dt / 2))

        _, dk3 = jacobian(self.model, (t + 0.5 * self.dt,
                                       state + 0.5 * k2 * self.dt))
        dk3 = torch.einsum(
            op_string, dk3, (jacobian_identity + dk2 * self.dt / 2))
        _, dk4 = jacobian(self.model, (t + self.dt,
                                       state + k3 * self.dt))

        dk4 = torch.einsum(
            op_string, dk4, (jacobian_identity + dk3 * self.dt))

        DM = jacobian_identity + _one_sixth * self.dt * \
            (dk1 + 2 * dk2 + 2 * dk3 + dk4)

        return DM


class VJRK4Method(nn.Module):
    def __init__(self, model, dt):
        super(VJRK4Method, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, t, state, value):

        k1 = self.model(t, state)
        k2 = self.model(t - 0.5*self.dt, state - 0.5*self.dt * k1)
        k3 = self.model(t - 0.5*self.dt, state - 0.5*self.dt * k2)

        __temp1__,  (__temp2__, dk1) = vjp(self.model, (t, state), v=value)

        print(f"dk1.shape:{dk1.shape},state.shape:{state.shape}")

        __temp1__,  (__temp2__, dk2) = vjp(self.model, (t - 0.5 * self.dt,
                                                        state - 0.5 * k1 * self.dt), v=value)

        print(f"dk2.shape:{dk2.shape},state.shape:{state.shape}")

        __temp1__,  (__temp2__, dk3) = vjp(self.model, (t - 0.5 * self.dt,
                                                        state - 0.5 * k2 * self.dt), v=value)

        print(f"dk3.shape:{dk3.shape},state.shape:{state.shape}")

        __temp1__,  (__temp2__, dk4) = vjp(self.model, (t - self.dt,
                                                        state - k3 * self.dt), v=value)
        print(f"dk4.shape:{dk4.shape},state.shape:{state.shape}")

        DM = self.dt / 6.0 * (dk1 + 2 * dk2 + 2 * dk3 + dk4)

        return DM
