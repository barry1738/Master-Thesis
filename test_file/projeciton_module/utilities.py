import torch
from torch.func import functional_call, vjp
# torch.set_default_dtype(torch.float64)


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

def qr_decomposition(J_mat, diff, mu, device):
    """Solve the linear system using QR decomposition"""
    A = torch.vstack((J_mat, mu**0.5 * torch.eye(J_mat.size(1), device=device)))
    b = torch.vstack((-diff, torch.zeros(J_mat.size(1), 1, device=device)))
    Q, R = torch.linalg.qr(A)
    x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
    return x.flatten()


def cholesky(J, diff, mu, device):
    """Solve the linear system using Cholesky decomposition"""
    A = J.t() @ J + mu * torch.eye(J.shape[1], device=device)
    b = J.t() @ -diff
    L = torch.linalg.cholesky(A)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    x = torch.linalg.solve_triangular(L.t(), y, upper=True)
    return x.flatten()


def predict(model, params, x, y):
    """Predict the model output."""
    return functional_call(model, params, (x, y))
    
def predict_dx(model, params, x, y):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
    return vjpfunc(torch.ones_like(output))[0]

def predict_dy(model, params, x, y):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
    return vjpfunc(torch.ones_like(output))[0]

def predict_dxx(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: predict_dx(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]

def predict_dxy(model, params, x, y):
    """Compute the mixed directional derivative of the model output with respect to x and y."""
    output, vjpfunc = vjp(lambda primal: predict_dx(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]

def predict_dyx(model, params, x, y):
    """Compute the mixed directional derivative of the model output with respect to y and x."""
    output, vjpfunc = vjp(lambda primal: predict_dy(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]

def predict_dyy(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: predict_dy(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]