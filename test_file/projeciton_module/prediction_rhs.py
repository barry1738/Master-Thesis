import torch
import model_func as mf
from utilities import exact_sol
from torch.func import grad, vmap
from config import REYNOLDS_NUM, TIME_STEP

def predict_rhs(model, x, y, step, device):
    """
    Predict the right-hand side
    """
    if step == 2:
        u1 = exact_sol(x, y, TIME_STEP, REYNOLDS_NUM, "u")
        v1 = exact_sol(x, y, TIME_STEP, REYNOLDS_NUM, "v")
        u0 = exact_sol(x, y, 0.0, REYNOLDS_NUM, "u")
        v0 = exact_sol(x, y, 0.0, REYNOLDS_NUM, "v")
        du1dx = vmap(
            grad(exact_sol, argnums=0), 
            in_dims=(0, 0, None, None, None), 
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "u")
        du1dy = vmap(
            grad(exact_sol, argnums=1),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "u")
        dv1dx = vmap(
            grad(exact_sol, argnums=0),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "v")
        dv1dy = vmap(
            grad(exact_sol, argnums=1),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "v")
        dp1dx = vmap(
            grad(exact_sol, argnums=0),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "p")
        dp1dy = vmap(
            grad(exact_sol, argnums=1),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), TIME_STEP, REYNOLDS_NUM, "p")
        du0dx = vmap(
            grad(exact_sol, argnums=0),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), 0.0, REYNOLDS_NUM, "u")
        du0dy = vmap(
            grad(exact_sol, argnums=1),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), 0.0, REYNOLDS_NUM, "u")
        dv0dx = vmap(
            grad(exact_sol, argnums=0),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), 0.0, REYNOLDS_NUM, "v")
        dv0dy = vmap(
            grad(exact_sol, argnums=1),
            in_dims=(0, 0, None, None, None),
            out_dims=0
            )(x.reshape(-1), y.reshape(-1), 0.0, REYNOLDS_NUM, "v")

        rhs_u = (
            4 * u1 - u0
            - 2 * (2 * TIME_STEP) * (u1 * du1dx + v1 * du1dy)
            + (2 * TIME_STEP) * (u0 * du0dx + v0 * du0dy)
            - (2 * TIME_STEP) * dp1dx
        ).to(device)

        rhs_v = (
            4 * v1 - v0
            - 2 * (2 * TIME_STEP) * (u1 * dv1dx + v1 * dv1dy)
            + (2 * TIME_STEP) * (u0 * dv0dx + v0 * dv0dy)
            - (2 * TIME_STEP) * dp1dy
        ).to(device)

        rhs_u_bd = exact_sol(x, y, step * TIME_STEP, REYNOLDS_NUM, "u")
        rhs_v_bd = exact_sol(x, y, step * TIME_STEP, REYNOLDS_NUM, "v")

        return rhs_u, rhs_v, rhs_u_bd, rhs_v_bd

    elif step == 3:
        