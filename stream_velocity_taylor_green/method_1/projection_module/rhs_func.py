import torch
from projection_module.utilities import exact_sol


def prediction_step_rhs(x, y, prev_val, step, Dt, Re, device):
    x_inner, x_bd = x
    y_inner, y_bd = y

    Rf_u_inner = (
        4 * torch.tensor(prev_val["u1"][:x_inner.size(0), :])
        - torch.tensor(prev_val["u0"][:x_inner.size(0), :])
        - 2 * (2 * Dt) * (torch.tensor(prev_val["u1"][:x_inner.size(0), :]) * 
                          torch.tensor(prev_val["du1dx"][:x_inner.size(0), :]) + 
                          torch.tensor(prev_val["v1"][:x_inner.size(0), :]) * 
                          torch.tensor(prev_val["du1dy"][:x_inner.size(0), :]))
        + (2 * Dt) * (torch.tensor(prev_val["u0"][:x_inner.size(0), :]) * 
                      torch.tensor(prev_val["du0dx"][:x_inner.size(0), :]) + 
                      torch.tensor(prev_val["v0"][:x_inner.size(0), :]) * 
                      torch.tensor(prev_val["du0dy"][:x_inner.size(0), :]))
        - (2 * Dt) * (torch.tensor(prev_val["dp1dx"][:x_inner.size(0), :]))
    ).to(device)

    Rf_v_inner = (
        4 * torch.tensor(prev_val["v1"][:x_inner.size(0), :])
        - torch.tensor(prev_val["v0"][:x_inner.size(0), :])
        - 2 * (2 * Dt) * (torch.tensor(prev_val["u1"][:x_inner.size(0), :]) * 
                          torch.tensor(prev_val["dv1dx"][:x_inner.size(0), :]) + 
                          torch.tensor(prev_val["v1"][:x_inner.size(0), :]) * 
                          torch.tensor(prev_val["dv1dy"][:x_inner.size(0), :]))
        + (2 * Dt) * (torch.tensor(prev_val["u0"][:x_inner.size(0), :]) * 
                      torch.tensor(prev_val["dv0dx"][:x_inner.size(0), :]) + 
                      torch.tensor(prev_val["v0"][:x_inner.size(0), :]) * 
                      torch.tensor(prev_val["dv0dy"][:x_inner.size(0), :]))
        - (2 * Dt) * (torch.tensor(prev_val["dp1dy"][:x_inner.size(0), :]))
    ).to(device)

    Rf_u_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "u").to(device)
    Rf_v_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "v").to(device)

    return Rf_u_inner, Rf_v_inner, Rf_u_bd, Rf_v_bd