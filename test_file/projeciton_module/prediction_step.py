import torch
import torch.nn as nn
from torch.func import vmap, jacrev, grad
from config import REYNOLDS_NUM, TIME_STEP
from mesh_generator import CreateSquareMesh, CreateCircleMesh
from utilities import exact_sol, qr_decomposition, cholesky
from model_func import predict, predict_dxx, predict_dyy


Re = REYNOLDS_NUM
Dt = TIME_STEP
# print(f"Re = {REYNOLDS_NUM}")
# print(f"Dt = {TIME_STEP}")


def prediction_step(model, rhs_vec, step, direction, device):
    """The prediction step of the projection method"""
    # Create the mesh
    mesh = CreateSquareMesh()
    x_inner, y_inner = mesh.inner_points(100)
    x_bd, y_bd = mesh.boundary_points(10)
    x_inner_v, y_inner_v = mesh.inner_points(1000)
    x_bd_v, y_bd_v = mesh.boundary_points(100)

    # Move the data to the device
    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
    x_inner_v, y_inner_v = x_inner_v.to(device), y_inner_v.to(device)
    x_bd_v, y_bd_v = x_bd_v.to(device), y_bd_v.to(device)

    # Define the parameters
    def weights_init(model):
        """Initialize the weights of the neural network."""
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight.data, gain=5)
            # nn.init.xavier_normal_(model.weight.data, gain=5)

    model.apply(weights_init)
    params = model.state_dict()

    # Unpack the right-hand side values
    Rf_inner = rhs_vec[0]
    Rf_bd = rhs_vec[1]
    Rf_inner_valid = rhs_vec[2]
    Rf_bd_valid = rhs_vec[3]

    # Compute the right-hand side values
    if direction == "u":
        """Compute the right-hand side values for the u-component."""
        Rf_inner = (
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

        Rf_inner_valid = (
            4 * torch.tensor(prev_val_valid["u1"][:x_inner_v.size(0), :])
            - torch.tensor(prev_val_valid["u0"][:x_inner_v.size(0), :])
            - 2 * (2 * Dt) * (torch.tensor(prev_val_valid["u1"][:x_inner_v.size(0), :]) * 
                            torch.tensor(prev_val_valid["du1dx"][:x_inner_v.size(0), :]) + 
                            torch.tensor(prev_val_valid["v1"][:x_inner_v.size(0), :]) * 
                            torch.tensor(prev_val_valid["du1dy"][:x_inner_v.size(0), :]))
            + (2 * Dt) * (torch.tensor(prev_val_valid["u0"][:x_inner_v.size(0), :]) * 
                        torch.tensor(prev_val_valid["du0dx"][:x_inner_v.size(0), :]) + 
                        torch.tensor(prev_val_valid["v0"][:x_inner_v.size(0), :]) * 
                        torch.tensor(prev_val_valid["du0dy"][:x_inner_v.size(0), :]))
            - (2 * Dt) * (torch.tensor(prev_val_valid["dp1dx"][:x_inner_v.size(0), :]))
        ).to(device)

        Rf_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "u")
        Rf_bd_valid = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "u")

    else:
        """Compute the right-hand side values for the v-component."""
        Rf_inner = (
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

        Rf_inner_valid = (
            4 * torch.tensor(prev_val_valid["v1"][:x_inner_v.size(0), :])
            - torch.tensor(prev_val_valid["v0"][:x_inner_v.size(0), :])
            - 2 * (2 * Dt) * (torch.tensor(prev_val_valid["u1"][:x_inner_v.size(0), :]) * 
                            torch.tensor(prev_val_valid["dv1dx"][:x_inner_v.size(0), :]) + 
                            torch.tensor(prev_val_valid["v1"][:x_inner_v.size(0), :]) * 
                            torch.tensor(prev_val_valid["dv1dy"][:x_inner_v.size(0), :]))
            + (2 * Dt) * (torch.tensor(prev_val_valid["u0"][:x_inner_v.size(0), :]) * 
                        torch.tensor(prev_val_valid["dv0dx"][:x_inner_v.size(0), :]) + 
                        torch.tensor(prev_val_valid["v0"][:x_inner_v.size(0), :]) * 
                        torch.tensor(prev_val_valid["dv0dy"][:x_inner_v.size(0), :]))
            - (2 * Dt) * (torch.tensor(prev_val_valid["dp1dy"][:x_inner_v.size(0), :]))
        ).to(device)

        Rf_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "v")
        Rf_bd_valid = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "v")


    def compute_loss_res(model, params, x, y, Rf_inner):
        """Compute the residual loss function."""
        pred = (
            3 * predict(model, params, x, y) -
            (2 * Dt / Re) * (
                predict_dxx(model, params, x, y) +
                predict_dyy(model, params, x, y)
            )
        )
        loss_res = pred - Rf_inner
        return loss_res

    def compute_loss_bd(model, params, x, y, Rf_bd):
        """Compute the boundary loss function."""
        pred = predict(model, params, x, y)
        loss_bd = pred - Rf_bd
        return loss_bd
    
    # Start training
    Niter = 1000
    tol = 1.0e-9
    mu = 1.0e3
    alpha = 1.0
    beta = 1.0
    savedloss = []
    savedloss_valid = []

    for iter in range(Niter):
        # Compute the jacobi matrix
        jac_res_dict = vmap(
            jacrev(compute_loss_res, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
        )(model, params, x_inner, y_inner, Rf_inner)

        jac_bd_dict = vmap(
            jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
        )(model, params, x_bd, y_bd, Rf_bd)

        # Stack the jacobian matrix
        jac_res = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict.values()])
        jac_bd = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict.values()])
        jac_res *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        jac_bd *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))

        # Compute the residual of the loss function
        l_vec_res = compute_loss_res(model, params, x_inner, y_inner, Rf_inner)
        l_vec_bd = compute_loss_bd(model, params, x_bd, y_bd, Rf_bd)
        l_vec_res_valid = compute_loss_res(model, params, x_inner_v, y_inner_v, Rf_inner_valid)
        l_vec_bd_valid = compute_loss_bd(model, params, x_bd_v, y_bd_v, Rf_bd_valid)
        l_vec_res *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        l_vec_bd *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        l_vec_res_valid /= torch.sqrt(torch.tensor(x_inner_v.size(0)))
        l_vec_bd_valid /= torch.sqrt(torch.tensor(x_bd_v.size(0)))

        # Cat the Jacobian matrix and the loss function
        jacobian = torch.vstack((jac_res, jac_bd))
        l_vec = torch.vstack((l_vec_res, l_vec_bd))

        # Solve the non-linear system
        # p = cholesky(jacobian, l_vec, mu, device)
        p = qr_decomposition(jacobian, l_vec, mu, device)
        # Update the parameters
        params_flatten = nn.utils.parameters_to_vector(params.values())
        params_flatten += p
        nn.utils.vector_to_parameters(params_flatten, params.values())

        # Compute the loss function
        loss = torch.sum(l_vec_res**2) + torch.sum(l_vec_bd**2)
        loss_valid = torch.sum(l_vec_res_valid**2) + torch.sum(l_vec_bd_valid**2)
        savedloss.append(loss.item())
        savedloss_valid.append(loss_valid.item())

        # Stop the training if the loss function is converged
        if (iter == Niter - 1) or (loss < tol):
            print('Successful training u* ...')
            break

        # Update the parameter mu
        if iter % 3 == 0:
            if savedloss[iter] > savedloss[iter - 1]:
                mu = min(2 * mu, 1e8)
            else:
                mu = max(mu / 5, 1e-10)

        # Compute alpha_bar and beta_bar, then update alpha and beta
        if iter % 100 == 0:
            dloss_res_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_res(model, primal, x_inner, y_inner, Rf_inner) ** 2
                ),
                argnums=0,
            )(params)

            dloss_bd_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_bd(model, primal, x_bd, y_bd, Rf_bd) ** 2
                ),
                argnums=0,
            )(params)

            dloss_res_dp_flatten = nn.utils.parameters_to_vector(
                dloss_res_dp.values()
            ) / torch.tensor(x_inner.size(0))
            dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)

            dloss_bd_dp_flatten = nn.utils.parameters_to_vector(
                dloss_bd_dp.values()
            ) / torch.tensor(x_bd.size(0))
            dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)

            alpha_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_res_dp_norm
            beta_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_bd_dp_norm

            alpha = (1 - 0.1) * alpha + 0.1 * alpha_bar
            beta = (1 - 0.1) * beta + 0.1 * beta_bar
            print(f"alpha: {alpha:.2f}, beta: {beta:.2f}")

    return params, loss, loss_valid