# import torch
import model_func as mf
from torch.func import grad, vmap
from projection_module.utilities import exact_sol
from projection_module.config import REYNOLDS_NUM, TIME_STEP


def prediction_rhs(model, params, x, y, step, device):
    """
    Predict the right-hand side
    """
    (
        model_u_star,
        model_v_star,
        model_phi,
        model_p
    ) = model
    (
        params_u_star,
        params_v_star,
        params_phi,
        params_p,
        params_u_star_old,
        params_v_star_old,
        params_phi_old
    ) = params
    x_inner, x_bd = x
    y_inner, y_bd = y

    if step == 2:
        u1 = exact_sol(x_inner, y_inner, (step - 1) * TIME_STEP, REYNOLDS_NUM, "u")
        v1 = exact_sol(x_inner, y_inner, (step - 1) * TIME_STEP, REYNOLDS_NUM, "v")
        u0 = exact_sol(x_inner, y_inner, (step - 2) * TIME_STEP, REYNOLDS_NUM, "u")
        v0 = exact_sol(x_inner, y_inner, (step - 2) * TIME_STEP, REYNOLDS_NUM, "v")
        du1dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        du1dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        dv1dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)
        dv1dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)
        dp1dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "p").reshape(-1, 1)
        dp1dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 1) * TIME_STEP, REYNOLDS_NUM, "p").reshape(-1, 1)
        du0dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        du0dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        dv0dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)
        dv0dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)

        rhs_u = (
            4 * u1
            - u0
            - 2 * (2 * TIME_STEP) * (u1 * du1dx + v1 * du1dy)
            + (2 * TIME_STEP) * (u0 * du0dx + v0 * du0dy)
            - (2 * TIME_STEP) * dp1dx
        ).to(device)

        rhs_v = (
            4 * v1
            - v0
            - 2 * (2 * TIME_STEP) * (u1 * dv1dx + v1 * dv1dy)
            + (2 * TIME_STEP) * (u0 * dv0dx + v0 * dv0dy)
            - (2 * TIME_STEP) * dp1dy
        ).to(device)

        rhs_u_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "u")
        rhs_v_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "v")

        return rhs_u, rhs_v, rhs_u_bd, rhs_v_bd

    elif step == 3:
        u1 = mf.predict(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dx(model_phi, params_phi, x_inner, y_inner)
        v1 = mf.predict(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dy(model_phi, params_phi, x_inner, y_inner)
        u0 = exact_sol(x_inner, y_inner, (step - 2) * TIME_STEP, REYNOLDS_NUM, "u")
        v0 = exact_sol(x_inner, y_inner, (step - 2) * TIME_STEP, REYNOLDS_NUM, "v")
        du1dx = mf.predict_dx(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxx(model_phi, params_phi, x_inner, y_inner)
        du1dy = mf.predict_dy(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxy(model_phi, params_phi, x_inner, y_inner)
        dv1dx = mf.predict_dx(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyx(model_phi, params_phi, x_inner, y_inner)
        dv1dy = mf.predict_dy(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyy(model_phi, params_phi, x_inner, y_inner)
        dp1dx = mf.predict_dx(model_p, params_p, x_inner, y_inner)
        dp1dy = mf.predict_dy(model_p, params_p, x_inner, y_inner)
        du0dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        du0dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "u").reshape(-1, 1)
        dv0dx = vmap(
            grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)
        dv0dy = vmap(
            grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
        )(x_inner.reshape(-1), y_inner.reshape(-1), (step - 2) * TIME_STEP, REYNOLDS_NUM, "v").reshape(-1, 1)

        rhs_u = (
            4 * u1
            - u0
            - 2 * (2 * TIME_STEP) * (u1 * du1dx + v1 * du1dy)
            + (2 * TIME_STEP) * (u0 * du0dx + v0 * du0dy)
            - (2 * TIME_STEP) * dp1dx
        ).to(device)

        rhs_v = (
            4 * v1
            - v0
            - 2 * (2 * TIME_STEP) * (u1 * dv1dx + v1 * dv1dy)
            + (2 * TIME_STEP) * (u0 * dv0dx + v0 * dv0dy)
            - (2 * TIME_STEP) * dp1dy
        ).to(device)

        rhs_u_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "u")
        rhs_v_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "v")

        return rhs_u, rhs_v, rhs_u_bd, rhs_v_bd
    
    else:
        u1 = mf.predict(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dx(model_phi, params_phi, x_inner, y_inner)
        v1 = mf.predict(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dy(model_phi, params_phi, x_inner, y_inner)
        u0 = mf.predict(model_u_star, params_u_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dx(model_phi, params_phi, x_inner, y_inner)
        v0 = mf.predict(model_v_star, params_v_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dy(model_phi, params_phi, x_inner, y_inner)
        du1dx = mf.predict_dx(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxx(model_phi, params_phi, x_inner, y_inner)
        du1dy = mf.predict_dy(model_u_star, params_u_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxy(model_phi, params_phi, x_inner, y_inner)
        dv1dx = mf.predict_dx(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyx(model_phi, params_phi, x_inner, y_inner)
        dv1dy = mf.predict_dy(model_v_star, params_v_star, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyy(model_phi, params_phi, x_inner, y_inner)
        dp1dx = mf.predict_dx(model_p, params_p, x_inner, y_inner)
        dp1dy = mf.predict_dy(model_p, params_p, x_inner, y_inner)
        du0dx = mf.predict_dx(model_u_star, params_u_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxx(model_phi, params_phi_old, x_inner, y_inner)
        du0dy = mf.predict_dy(model_u_star, params_u_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dxy(model_phi, params_phi_old, x_inner, y_inner)
        dv0dx = mf.predict_dx(model_v_star, params_v_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyx(model_phi, params_phi_old, x_inner, y_inner)
        dv0dy = mf.predict_dy(model_v_star, params_v_star_old, x_inner, y_inner) - (
            2 * TIME_STEP / 3
        ) * mf.predict_dyy(model_phi, params_phi_old, x_inner, y_inner)

        rhs_u = (
            4 * u1
            - u0
            - 2 * (2 * TIME_STEP) * (u1 * du1dx + v1 * du1dy)
            + (2 * TIME_STEP) * (u0 * du0dx + v0 * du0dy)
            - (2 * TIME_STEP) * dp1dx
        ).to(device)

        rhs_v = (
            4 * v1
            - v0
            - 2 * (2 * TIME_STEP) * (u1 * dv1dx + v1 * dv1dy)
            + (2 * TIME_STEP) * (u0 * dv0dx + v0 * dv0dy)
            - (2 * TIME_STEP) * dp1dy
        ).to(device)

        rhs_u_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "u")
        rhs_v_bd = exact_sol(x_bd, y_bd, step * TIME_STEP, REYNOLDS_NUM, "v")

        return rhs_u, rhs_v, rhs_u_bd, rhs_v_bd