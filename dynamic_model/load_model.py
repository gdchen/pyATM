import numpy as np
from dynamic_model.Lorenz import Lorenz63, Lorenz96, JLorenz63, JLorenz96


def load_Lorenz63():
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
    u0True = np.array([1.0, 1.0, 1.0])
    dynamic_model = Lorenz63
    return dynamic_model, u0True, [sigma, beta, rho], JLorenz63


def load_Lorenz96():
    F = 8  # forcing term
    n = 36  # dimension of state
    u0 = F * np.ones(n)  # Initial state (equilibrium)
    u0[19] = u0[19] + 0.01  # Add small perturbation to 20th variable
    u0True = u0
    dynamic_model = Lorenz96
    return dynamic_model, u0True, [F], JLorenz96

# need some unit test
