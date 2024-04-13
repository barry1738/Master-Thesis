import torch
import torch.nn as nn
from torch.func import vmap, jacrev, grad
from projeciton_module.config import REYNOLDS_NUM, TIME_STEP
from projeciton_module.utilities import qr_decomposition, cholesky
from model_func import predict, predict_dxx, predict_dyy


def prediction_step(model, points, rhs_vec, device):
    """The prediction step of the projection method"""
    # Define the parameters
    def weights_init(model):
        """Initialize the weights of the neural network."""
        if isinstance(model, nn.Linear):
            # nn.init.xavier_uniform_(model.weight.data, gain=5)
            nn.init.xavier_normal_(model.weight.data, gain=5)

    model.apply(weights_init)
    params = model.state_dict()

    # Unpack the points
    x_inner = points[0]
    y_inner = points[1]
    x_bd = points[2]
    y_bd = points[3]
    x_inner_v = points[4]
    y_inner_v = points[5]
    x_bd_v = points[6]
    y_bd_v = points[7]

    # Unpack the right-hand side values
    Rf_inner = rhs_vec[0]
    Rf_bd = rhs_vec[1]
    Rf_inner_valid = rhs_vec[2]
    Rf_bd_valid = rhs_vec[3]

    def compute_loss_res(model, params, x, y, Rf_inner):
        """Compute the residual loss function."""
        pred = (
            3 * predict(model, params, x, y) -
            (2 * TIME_STEP / REYNOLDS_NUM) * (
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
            print('Successful training. ...')
            print(f"iter: {iter}, Loss: {loss.item():.2e}, mu: {mu:.1e}")
            break

        # Update the parameter mu
        if iter % 5 == 0:
            if savedloss[iter] > savedloss[iter - 1]:
                mu = min(2 * mu, 1e8)
            else:
                mu = max(mu / 3, 1e-12)

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

        if iter % 5 == 0:
            print(f"iter: {iter}, Loss: {loss.item():.2e}, mu: {mu:.1e}")

    return params, savedloss, savedloss_valid