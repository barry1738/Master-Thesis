import torch

torch.set_default_dtype(torch.float64)


def exact_sol(x, y, t, Re, type):
    """calculate analytical solution
    u(x,y,t) = -cos(πx)sin(πy)exp(-2π²t/RE)
    v(x,y,t) =  sin(πx)cos(πy)exp(-2π²t/RE)
    p(x,y,t) = -0.25(cos(2πx)+cos(2πy))exp(-4π²t/RE)
    """
    match type:
        case "u":
            return (
                -torch.cos(torch.pi * x)
                * torch.sin(torch.pi * y)
                * torch.exp(torch.tensor(-2 * torch.pi**2 * t / Re))
            )
        case "v":
            return (
                torch.sin(torch.pi * x)
                * torch.cos(torch.pi * y)
                * torch.exp(torch.tensor(-2 * torch.pi**2 * t / Re))
            )
        case "p":
            return (
                -0.25
                * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))
                * torch.exp(torch.tensor(-4 * torch.pi**2 * t / Re))
            )
