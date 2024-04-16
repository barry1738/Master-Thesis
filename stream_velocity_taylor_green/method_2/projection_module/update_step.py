import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from projection_module.utilities import qr_decomposition
from model_func import predict


def update_step(model, points, rhs_vec, device):
    """The update step for velocity and pressure fields"""
    # Define the parameters
    def weights_init(model):
        """Initialize the weights of the neural network."""
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight.data, gain=5)
            # nn.init.xavier_normal_(model.weight.data, gain=5)

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

    x = torch.vstack((x_inner, x_bd))
    y = torch.vstack((y_inner, y_bd))
    x_v = torch.vstack((x_inner_v, x_bd_v))
    y_v = torch.vstack((y_inner_v, y_bd_v))

    # Unpack the rhs_vec
    Rf_res = rhs_vec[0]
    Rf_res_valid = rhs_vec[1]

    def compute_loss(model, params, x, y, rhs_vec):
        """Compute the residual loss function."""
        pred = predict(model, params, x, y)
        loss_res = pred - rhs_vec
        return loss_res

    # Start training
    Niter = 1000
    tol = 1.0e-9
    mu = 1.0e3
    savedloss = []
    savedloss_valid = []

    for iter in range(Niter):
        # Compute the jacobi matrix
        jac_res_dict = vmap(
            jacrev(compute_loss, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, params, x, y, Rf_res)

        # Stack the jacobian matrix
        jac_res = torch.hstack([v.view(x.size(0), -1) for v in jac_res_dict.values()])
        jac_res /= torch.sqrt(torch.tensor(x.size(0)))

        # Compute the residual of the loss function
        l_vec_res = compute_loss(model, params, x, y, Rf_res)
        l_vec_res_valid = compute_loss(model, params, x_v, y_v, Rf_res_valid)
        l_vec_res /= torch.sqrt(torch.tensor(x.size(0)))
        l_vec_res_valid /= torch.sqrt(torch.tensor(x_v.size(0)))

        # Cat the Jacobian matrix and the loss function
        jacobian = jac_res
        l_vec = l_vec_res

        # Solve the non-linear system
        # p = cholesky(jacobian, l_vec, mu, device)
        p = qr_decomposition(jacobian, l_vec, mu, device)
        # Update the parameters
        params_flatten = nn.utils.parameters_to_vector(params.values())
        params_flatten += p
        nn.utils.vector_to_parameters(params_flatten, params.values())

        # Compute the loss function
        loss = torch.sum(l_vec_res**2)
        loss_valid = torch.sum(l_vec_res_valid**2)
        savedloss.append(loss.item())
        savedloss_valid.append(loss_valid.item())

        if iter % 50 == 0:
            print(f"iter = {iter}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Stop the training if the loss function is converged
        if (iter == Niter - 1) or (loss < tol):
            print("Successful training. ...")
            print(f"iter = {iter}, loss = {loss.item():.2e}, mu = {mu:.1e}")
            break

        # Update the parameter mu
        if iter % 3 == 0:
            if savedloss[iter] > savedloss[iter - 1]:
                mu = min(2 * mu, 1e8)
            else:
                mu = max(mu / 3, 1e-10)

    return params, savedloss, savedloss_valid