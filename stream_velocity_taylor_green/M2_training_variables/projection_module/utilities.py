import torch


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