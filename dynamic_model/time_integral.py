# %%

import numpy as np


def Euler(rhs, state, dt, *args):
    k1 = rhs(state, *args)
    new_state = state + dt*k1
    return new_state


def RK4(rhs, state, dt, *args):
    k1 = rhs(state, *args)
    k2 = rhs(state+k1*dt/2.0, *args)
    k3 = rhs(state+k2*dt/2.0, *args)
    k4 = rhs(state+k3*dt, *args)

    new_state = state + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return new_state


# %%
def JEuler(rhs, Jrhs, state, dt, *args):

    n = len(state)
    dk1 = Jrhs(state, *args)
    DM = np.eye(n) + dt*dk1
    return DM


def JRK4(rhs, Jrhs, state, dt, *args):

    n = len(state)
    k1 = rhs(state, *args)
    k2 = rhs(state+k1*dt/2, *args)
    k3 = rhs(state+k2*dt/2, *args)

    dk1 = Jrhs(state, *args)
    dk2 = Jrhs(state+k1*dt/2, *args) @ (np.eye(n)+dk1*dt/2)
    dk3 = Jrhs(state+k2*dt/2, *args) @ (np.eye(n)+dk2*dt/2)
    dk4 = Jrhs(state+k3*dt, *args) @ (np.eye(n)+dk3*dt)

    DM = np.eye(n) + (dt/6.0) * (dk1+2*dk2+2*dk3+dk4)
    return DM
