import torch
import torch.nn as nn
from torch.func import vmap, jacrev, grad
from projeciton_module.config import TIME_STEP
from projeciton_module.utilities import qr_decomposition, cholesky
from model_func import predict_dx, predict_dy


def projection_step(phi_model, psi_model, points, rhs_vec, device):
    """The projection step of the projection method"""
    # Define the parameters
    def weights_init(model):
        """Initialize the weights of the neural network."""
        if isinstance(model, nn.Linear):
            # nn.init.xavier_uniform_(model.weight.data, gain=10)
            nn.init.xavier_normal_(model.weight.data, gain=10)

    phi_model.apply(weights_init)
    psi_model.apply(weights_init)
    phi_params = phi_model.state_dict()
    psi_params = psi_model.state_dict()
    num_params_phi = phi_model.num_total_params()

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
    Rf_proj_1 = rhs_vec[0]
    Rf_proj_2 = rhs_vec[1]
    Rf_bd_1 = rhs_vec[2]
    Rf_bd_2 = rhs_vec[3]
    Rf_proj_1_valid = rhs_vec[4]
    Rf_proj_2_valid = rhs_vec[5]
    Rf_bd_1_valid = rhs_vec[6]
    Rf_bd_2_valid = rhs_vec[7]
    
    def compute_loss_res_1(model1, model2, params1, params2, x, y, Rf_proj_1):
        """Compute the residual loss function."""
        pred = (
            predict_dy(model2, params2, x, y) 
            + (2 * TIME_STEP / 3) * predict_dx(model1, params1, x, y)
        )
        loss_res = pred - Rf_proj_1
        return loss_res

    def compute_loss_res_2(model1, model2, params1, params2, x, y, Rf_proj_2):
        """Compute the residual loss function."""
        pred = (
            - predict_dx(model2, params2, x, y) 
            + (2 * TIME_STEP / 3) * predict_dy(model1, params1, x, y)
        )
        loss_res = pred - Rf_proj_2
        return loss_res

    def compute_loss_bd_1(model1, model2, params1, params2, x, y, Rf_bd_1):
        """Compute the boundary loss function."""
        pred = predict_dy(model2, params2, x, y)
        loss_bd = pred - Rf_bd_1
        return loss_bd

    def compute_loss_bd_2(model1, model2, params1, params2, x, y, Rf_bd_2):
        """Compute the boundary loss function."""
        pred = -predict_dx(model2, params2, x, y)
        loss_bd = pred - Rf_bd_2
        return loss_bd

    # Start training
    Niter = 1000
    tol = 1.0e-8
    mu = 1.0e3
    alpha = 1.0
    beta = 1.0
    savedloss = []
    savedloss_valid = []

    for iter in range(Niter):
        # Compute the jacobi matrix
        jac_res_1_phi_dict, jac_res_1_psi_dict = vmap(
            jacrev(compute_loss_res_1, argnums=(2, 3)),
            in_dims=(None, None, None, None, 0, 0, 0),
            out_dims=0,
        )(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_proj_1)

        jac_res_2_phi_dict, jac_res_2_psi_dict = vmap(
            jacrev(compute_loss_res_2, argnums=(2, 3)),
            in_dims=(None, None, None, None, 0, 0, 0),
            out_dims=0,
        )(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_proj_2)

        jac_bd_1_phi_dict, jac_bd_1_psi_dict = vmap(
            jacrev(compute_loss_bd_1, argnums=(2, 3)),
            in_dims=(None, None, None, None, 0, 0, 0),
            out_dims=0,
        )(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, Rf_bd_1)

        jac_bd_2_phi_dict, jac_bd_2_psi_dict = vmap(
            jacrev(compute_loss_bd_2, argnums=(2, 3)),
            in_dims=(None, None, None, None, 0, 0, 0),
            out_dims=0,
        )(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, Rf_bd_2)

        # Stack the jacobian matrix
        jac_res_1_phi = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_1_phi_dict.values()])
        jac_res_1_psi = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_1_psi_dict.values()])
        jac_res_2_phi = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_2_phi_dict.values()])
        jac_res_2_psi = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_2_psi_dict.values()])
        jac_bd_1_phi = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_1_phi_dict.values()])
        jac_bd_1_psi = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_1_psi_dict.values()])
        jac_bd_2_phi = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_2_phi_dict.values()])
        jac_bd_2_psi = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_2_psi_dict.values()])
        jac_res_1_phi *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        jac_res_1_psi *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        jac_res_2_phi *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        jac_res_2_psi *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        jac_bd_1_phi *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        jac_bd_1_psi *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        jac_bd_2_phi *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        jac_bd_2_psi *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))

        # Compute the residual of the loss function
        l_vec_res_1 = compute_loss_res_1(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_proj_1)
        l_vec_res_2 = compute_loss_res_2(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_proj_2)
        l_vec_bd_1 = compute_loss_bd_1(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, Rf_bd_1)
        l_vec_bd_2 = compute_loss_bd_2(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, Rf_bd_2)
        l_vec_res_1_valid = compute_loss_res_1(phi_model, psi_model, phi_params, psi_params, x_inner_v, y_inner_v, Rf_proj_1_valid)
        l_vec_res_2_valid = compute_loss_res_2(phi_model, psi_model, phi_params, psi_params, x_inner_v, y_inner_v, Rf_proj_2_valid)
        l_vec_bd_1_valid = compute_loss_bd_1(phi_model, psi_model, phi_params, psi_params, x_bd_v, y_bd_v, Rf_bd_1_valid)
        l_vec_bd_2_valid = compute_loss_bd_2(phi_model, psi_model, phi_params, psi_params, x_bd_v, y_bd_v, Rf_bd_2_valid)
        l_vec_res_1 *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        l_vec_res_2 *= torch.sqrt(alpha / torch.tensor(x_inner.size(0)))
        l_vec_bd_1 *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        l_vec_bd_2 *= torch.sqrt(beta / torch.tensor(x_bd.size(0)))
        l_vec_res_1_valid /= torch.sqrt(torch.tensor(x_inner_v.size(0)))
        l_vec_res_2_valid /= torch.sqrt(torch.tensor(x_inner_v.size(0)))
        l_vec_bd_1_valid /= torch.sqrt(torch.tensor(x_bd_v.size(0)))
        l_vec_bd_2_valid /= torch.sqrt(torch.tensor(x_bd_v.size(0)))

        # Cat the Jacobian matrix and the loss function
        jacobian = torch.vstack((
            torch.hstack((jac_res_1_phi, jac_res_1_psi)),
            torch.hstack((jac_res_2_phi, jac_res_2_psi)),
            torch.hstack((jac_bd_1_phi, jac_bd_1_psi)),
            torch.hstack((jac_bd_2_phi, jac_bd_2_psi))
        ))
        l_vec = torch.vstack((l_vec_res_1, l_vec_res_2, l_vec_bd_1, l_vec_bd_2))

        # Solve the non-linear system
        # p = cholesky(jacobian, l_vec, mu, device)
        p = qr_decomposition(jacobian, l_vec, mu, device)
        # Update the parameters
        phi_params_flatten = nn.utils.parameters_to_vector(phi_params.values())
        psi_params_flatten = nn.utils.parameters_to_vector(psi_params.values())
        phi_params_flatten += p[:num_params_phi]
        psi_params_flatten += p[num_params_phi:]
        nn.utils.vector_to_parameters(phi_params_flatten, phi_params.values())
        nn.utils.vector_to_parameters(psi_params_flatten, psi_params.values())

        # Compute the loss function
        loss = torch.sum(l_vec_res_1**2 + l_vec_res_2**2) + torch.sum(
            l_vec_bd_1**2 + l_vec_bd_2**2
        )
        loss_valid = torch.sum(
            l_vec_res_1_valid**2 + l_vec_res_2_valid**2
        ) + torch.sum(l_vec_bd_1_valid**2 + l_vec_bd_2_valid**2)
        savedloss.append(loss.item())
        savedloss_valid.append(loss_valid.item())

        # Update alpha and beta
        if iter % 100 == 0:
            dloss_res_1_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_res_1(phi_model, psi_model, primal, psi_params, x_inner, y_inner, Rf_proj_1)** 2
                ), argnums=0,
            )(phi_params)

            dloss_res_2_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_res_2(phi_model, psi_model, phi_params, primal, x_inner, y_inner, Rf_proj_2) ** 2
                ), argnums=0,
            )(phi_params)

            dloss_bd_1_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_bd_1(phi_model, psi_model, primal, psi_params, x_bd, y_bd, Rf_bd_1) ** 2
                ), argnums=0,
            )(phi_params)

            dloss_bd_2_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_bd_2(phi_model, psi_model, phi_params, primal, x_bd, y_bd, Rf_bd_2) ** 2
                ), argnums=0,
            )(phi_params)

            dloss_res_1_dp_flatten = nn.utils.parameters_to_vector(
                dloss_res_1_dp.values()
            ) / torch.tensor(2 * x_inner.size(0))
            dloss_res_2_dp_flatten = nn.utils.parameters_to_vector(
                dloss_res_2_dp.values()
            ) / torch.tensor(2 * x_inner.size(0))
            dloss_res_dp_flatten = torch.hstack(
                (dloss_res_1_dp_flatten, dloss_res_2_dp_flatten)
            )
            dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)

            dloss_bd_dp_1_flatten = nn.utils.parameters_to_vector(
                dloss_bd_1_dp.values()
            ) / torch.tensor(2 * x_bd.size(0))
            dloss_bd_dp_2_flatten = nn.utils.parameters_to_vector(
                dloss_bd_2_dp.values()
            ) / torch.tensor(2 * x_bd.size(0))
            dloss_bd_dp_flatten = torch.hstack(
                (dloss_bd_dp_1_flatten, dloss_bd_dp_2_flatten)
            )
            dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)

            alpha_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_res_dp_norm
            beta_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_bd_dp_norm

            alpha = (1 - 0.1) * alpha + 0.1 * alpha_bar
            beta = (1 - 0.1) * beta + 0.1 * beta_bar
            print(f"alpha: {alpha:.2f}, beta: {beta:.2f}")

        if iter % 5 == 0:
            print(f"iter = {iter}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Stop the training if the loss function is converged
        if (iter == Niter - 1) or (loss < tol):
            print("Successful training. ...")
            print(f"iter = {iter}, loss = {loss.item():.2e}, mu = {mu:.1e}")
            break
        elif loss.item() > 1.0e10:
            print("Failed training. ...")
            print(f"iter: {iter}, Loss: {loss.item():.2e}, mu: {mu:.1e}")
            break

        # Update the parameter mu
        if iter % 3 == 0:
            if savedloss[iter] > savedloss[iter - 1]:
                mu = min(2 * mu, 1e8)
            else:
                mu = max(mu / 3, 1e-12)

    return phi_params, psi_params, savedloss, savedloss_valid