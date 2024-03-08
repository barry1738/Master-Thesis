import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)


def exact_solution(x, y, t, Re=1600):
    """
    Calculate analytical solution
    u(x,y,t) = -cos(πx)sin(πy)exp(-2π²t/RE)
    v(x,y,t) =  sin(πx)cos(πy)exp(-2π²t/RE)
    p(x,y,t) = -0.25(cos(2πx)+cos(2πy))exp(-4π²t/RE)
    """
    match type:
        case "u":
            exp_val = torch.tensor([-2 * torch.pi**2 * t / Re])
            func = (
                -torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(exp_val)
            )
            return func
        case "v":
            exp_val = torch.tensor([-2 * torch.pi**2 * t / Re])
            func = (
                torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(exp_val)
            )
            return func
        case "p":
            exp_val = torch.tensor([-4 * torch.pi**2 * t / Re])
            func = (
                -0.25
                * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))
                * torch.exp(exp_val)
            )
            return func