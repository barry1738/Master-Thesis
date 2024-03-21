"""
Taylor-Green Vortex flow using Physics-Informed Neural Networks (PINNs)

Incompressible Navier-Stokes Equations:
u_tt + u*u_x - v*u_y = - p_x - RE*Δu
v_tt + u*v_x + v*v_y = - p_y - RE*Δv
u_x + v_y = 0

Projection Method:
Step 1: Predict the intermediate velocity field (u*, v*) using the neural network
Step 2: Project the intermediate velocity field onto the space of divergence-free fields
Step 3: Update the velocity and pressure fields
"""


import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad
from mesh_generator import CreateMesh
from utilities_2 import exact_sol, qr_decomposition, cholesky


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define the neural network
class PinnModel(nn.Module):
    def __init__(self, layers):
        super(PinnModel, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x, y):
        """Forward pass of the neural network."""
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output
    
    def predict(self, model, params, x, y):
        """Predict the model output."""
        return functional_call(model, params, (x, y))
    
    def predict_dx(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dy(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dxx(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: self.predict_dx(model, params, primal, y), x)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dyy(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: self.predict_dy(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    

def prediction_step(model, training_data, prev_val, prev_val_valid, step):
    """The prediction step of the projection method"""
    # Unpack the training data
    x_inner, y_inner = training_data[0]
    x_bd, y_bd = training_data[1]
    x_inner_v, y_inner_v = training_data[2]
    x_bd_v, y_bd_v = training_data[3]
    nx, ny = training_data[4]

    # Define the parameters
    u_star_params = dict(model.named_parameters())
    v_star_params = dict(model.named_parameters())

    # Compute the right-hand side values
    Rf_u_inner = (
        4 * prev_val["u1"]
        - prev_val["u0"]
        - 2 * (2 * Dt) * (prev_val["u1"] * prev_val["du1dx"] + 
                          prev_val["v1"] * prev_val["du1dy"])
        + (2 * Dt) * (prev_val["u0"] * prev_val["du0dx"] + 
                      prev_val["v0"] * prev_val["du0dy"])
        - (2 * Dt) * (prev_val["dp1dx"])
    )
    Rf_v_inner = (
        4 * prev_val["v1"]
        - prev_val["v0"]
        - 2 * (2 * Dt) * (prev_val["u1"] * prev_val["dv1dx"] + 
                          prev_val["v1"] * prev_val["dv1dy"])
        + (2 * Dt) * (prev_val["u0"] * prev_val["dv0dx"] + 
                      prev_val["v0"] * prev_val["dv0dy"])
        - (2 * Dt) * (prev_val["dp1dy"])
    )
    Rf_u_inner_valid = (
        4 * prev_val_valid["u1"]
        - prev_val_valid["u0"]
        - 2 * (2 * Dt) * (prev_val_valid["u1"] * prev_val_valid["du1dx"] + 
                          prev_val_valid["v1"] * prev_val_valid["du1dy"])
        + (2 * Dt) * (prev_val_valid["u0"] * prev_val_valid["du0dx"] + 
                      prev_val_valid["v0"] * prev_val_valid["du0dy"])
        - (2 * Dt) * (prev_val_valid["dp1dx"])
    )
    Rf_v_inner_valid = (
        4 * prev_val_valid["v1"]
        - prev_val_valid["v0"]
        - 2 * (2 * Dt) * (prev_val_valid["u1"] * prev_val_valid["dv1dx"] + 
                          prev_val_valid["v1"] * prev_val_valid["dv1dy"])
        + (2 * Dt) * (prev_val_valid["u0"] * prev_val_valid["dv0dx"] + 
                      prev_val_valid["v0"] * prev_val_valid["dv0dy"])
        - (2 * Dt) * (prev_val_valid["dp1dy"])
    )
    Rf_u_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "u")
    Rf_v_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "v")
    Rf_u_bd_valid = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "u")
    Rf_v_bd_valid = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "v")


    def compute_loss_res(model, params, x, y, Rf_inner):
        """Compute the residual loss function."""
        pred = (
            3 * model.predict(model, params, x, y) -
            (2 * Dt / Re) * (
                model.predict_dxx(model, params, x, y) +
                model.predict_dyy(model, params, x, y)
            )
        )
        loss_res = pred - Rf_inner
        return loss_res

    def compute_loss_bd(model, params, x, y, Rf_bd):
        """Compute the boundary loss function."""
        pred = model.predict(model, params, x, y)
        loss_bd = pred - Rf_bd
        return loss_bd
    
    # Start training
    Niter = 1000
    tol = 1.0e-10
    mu_u = 1.0e3
    mu_v = 1.0e3
    alpha_u = 1.0
    alpha_v = 1.0
    beta_u = 1.0
    beta_v = 1.0
    savedloss_u = []
    savedloss_u_valid = []
    savedloss_v = []
    savedloss_v_valid = []
    u_star_finished = False
    v_star_finished = False

    for iter in range(Niter):
        if u_star_finished is not True:
            # Compute the jacobi matrix
            jac_res_dict_u = vmap(
                jacrev(compute_loss_res, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model, u_star_params, x_inner, y_inner, Rf_u_inner)

            jac_bd_dict_u = vmap(
                jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model, u_star_params, x_bd, y_bd, Rf_u_bd)

            # Stack the jacobian matrix
            jac_res_u = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict_u.values()])
            jac_bd_u = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict_u.values()])
            jac_res_u *= torch.sqrt(alpha_u / torch.tensor(x_inner.size(0)))
            jac_bd_u *= torch.sqrt(beta_u / torch.tensor(x_bd.size(0)))

            # Compute the residual of the loss function
            l_vec_res_u = compute_loss_res(model, u_star_params, x_inner, y_inner, Rf_u_inner)
            l_vec_bd_u = compute_loss_bd(model, u_star_params, x_bd, y_bd, Rf_u_bd)
            l_vec_res_u_valid = compute_loss_res(model, u_star_params, x_inner_v, y_inner_v, Rf_u_inner_valid)
            l_vec_bd_u_valid = compute_loss_bd(model, u_star_params, x_bd_v, y_bd_v, Rf_u_bd_valid)
            l_vec_res_u *= torch.sqrt(alpha_u / torch.tensor(x_inner.size(0)))
            l_vec_bd_u *= torch.sqrt(beta_u / torch.tensor(x_bd.size(0)))
            l_vec_res_u_valid /= torch.sqrt(torch.tensor(x_inner_v.size(0)))
            l_vec_bd_u_valid /= torch.sqrt(torch.tensor(x_bd_v.size(0)))

            # Cat the Jacobian matrix and the loss function
            jacobian_u = torch.vstack((jac_res_u, jac_bd_u))
            l_vec_u = torch.vstack((l_vec_res_u, l_vec_bd_u))

            # Solve the non-linear system
            p_u = cholesky(jacobian_u, l_vec_u, mu_u, device)
            # Update the parameters
            u_params_flatten = nn.utils.parameters_to_vector(u_star_params.values())
            u_params_flatten += p_u
            nn.utils.vector_to_parameters(u_params_flatten, u_star_params.values())

            # Compute the loss function
            loss_u = torch.sum(l_vec_res_u**2) + torch.sum(l_vec_bd_u**2)
            loss_u_valid = torch.sum(l_vec_res_u_valid**2) + torch.sum(l_vec_bd_u_valid**2)
            savedloss_u.append(loss_u.item())
            savedloss_u_valid.append(loss_u_valid.item())

            print(f"iter: {iter}, loss_u: {loss_u.item():.2e}, mu_u: {mu_u:.1e}")

            # Stop the training if the loss function is converged
            if (iter == Niter - 1) or (loss_u < tol):
                break

            # Update the parameter mu
            if iter % 3 == 0:
                if savedloss_u[iter] > savedloss_u[iter - 1]:
                    mu_u = min(2 * mu_u, 1e8)
                else:
                    mu_u = max(mu_u / 5, 1e-10)

            # Compute alpha_bar and beta_bar, then update alpha and beta
            if iter % 100 == 0:
                dloss_res_dp_u = grad(
                    lambda primal: torch.sum(
                        compute_loss_res(model, primal, x_inner, y_inner, Rf_u_inner) ** 2
                    ),
                    argnums=0,
                )(u_star_params)

                dloss_bd_dp_u = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd(model, primal, x_bd, y_bd, Rf_u_bd) ** 2
                    ),
                    argnums=0,
                )(u_star_params)

                dloss_res_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_res_dp_u.values()
                ) / torch.tensor(x_inner.size(0))
                dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)

                dloss_bd_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_bd_dp_u.values()
                ) / torch.tensor(x_bd.size(0))
                dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)

                alpha_u_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_res_dp_norm
                beta_u_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_bd_dp_norm

                alpha_u = (1 - 0.1) * alpha_u + 0.1 * alpha_u_bar
                beta_u = (1 - 0.1) * beta_u + 0.1 * beta_u_bar
                print(f"alpha_u: {alpha_u:.2f}, beta_u: {beta_u:.2f}")

        if v_star_finished is not True:
            # Compute the jacobi matrix
            jac_res_dict_v = vmap(
                jacrev(compute_loss_res, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model, v_star_params, x_inner, y_inner, Rf_v_inner)

            jac_bd_dict_v = vmap(
                jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model, v_star_params, x_bd, y_bd, Rf_v_bd)

            # Stack the jacobian matrix
            jac_res_v = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict_v.values()])
            jac_bd_v = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict_v.values()])
            jac_res_v *= torch.sqrt(alpha_v / torch.tensor(x_inner.size(0)))
            jac_bd_v *= torch.sqrt(beta_v / torch.tensor(x_bd.size(0)))

            # Compute the residual of the loss function
            l_vec_res_v = compute_loss_res(model, v_star_params, x_inner, y_inner, Rf_v_inner)
            l_vec_bd_v = compute_loss_bd(model, v_star_params, x_bd, y_bd, Rf_v_bd)
            l_vec_res_v_valid = compute_loss_res(model, v_star_params, x_inner_v, y_inner_v, Rf_v_inner_valid)
            l_vec_bd_v_valid = compute_loss_bd(model, v_star_params, x_bd_v, y_bd_v, Rf_v_bd_valid)
            l_vec_res_v *= torch.sqrt(alpha_v / torch.tensor(x_inner.size(0)))
            l_vec_bd_v *= torch.sqrt(beta_v / torch.tensor(x_bd.size(0)))
            l_vec_res_v_valid /= torch.sqrt(torch.tensor(x_inner_v.size(0)))
            l_vec_bd_v_valid /= torch.sqrt(torch.tensor(x_bd_v.size(0)))

            # Cat the Jacobian matrix and the loss function
            jacobian_v = torch.vstack((jac_res_v, jac_bd_v))
            l_vec_v = torch.vstack((l_vec_res_v, l_vec_bd_v))

            # Solve the non-linear system
            p_v = cholesky(jacobian_v, l_vec_v, mu_v, device)
            # Update the parameters
            v_params_flatten = nn.utils.parameters_to_vector(v_star_params.values())
            v_params_flatten += p_v
            nn.utils.vector_to_parameters(v_params_flatten, v_star_params.values())

            # Compute the loss function
            loss_v = torch.sum(l_vec_res_v**2) + torch.sum(l_vec_bd_v**2)
            loss_v_valid = torch.sum(l_vec_res_v_valid**2) + torch.sum(l_vec_bd_v_valid**2)
            savedloss_v.append(loss_v.item())
            savedloss_v_valid.append(loss_v_valid.item())

            print(f"iter: {iter}, loss_v: {loss_v.item():.2e}, mu_v: {mu_v:.1e}")

            # Stop the training if the loss function is converged
            if (iter == Niter - 1) or (loss_v < tol):
                break

            # Update the parameter mu
            if iter % 3 == 0:
                if savedloss_v[iter] > savedloss_v[iter - 1]:
                    mu_v = min(2 * mu_v, 1e8)
                else:
                    mu_v = max(mu_v / 5, 1e-10)

            # Compute alpha_bar and beta_bar, then update alpha and beta
            if iter % 100 == 0:
                dloss_res_dp_v = grad(
                    lambda primal: torch.sum(
                        compute_loss_res(model, primal, x_inner, y_inner, Rf_v_inner) ** 2
                    ),
                    argnums=0,
                )(v_star_params)

                dloss_bd_dp_v = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd(model, primal, x_bd, y_bd, Rf_v_bd) ** 2
                    ),
                    argnums=0,
                )(v_star_params)

                dloss_res_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_res_dp_v.values()
                ) / torch.tensor(x_inner.size(0))
                dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)

                dloss_bd_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_bd_dp_v.values()
                ) / torch.tensor(x_bd.size(0))
                dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)

                alpha_v_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_res_dp_norm
                beta_v_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_bd_dp_norm

                alpha_v = (1 - 0.1) * alpha_v + 0.1 * alpha_v_bar
                beta_v = (1 - 0.1) * beta_v + 0.1 * beta_v_bar
                print(f"alpha_v: {alpha_v:.2f}, beta_v: {beta_v:.2f}")

        if u_star_finished is not True and v_star_finished is not True:
            print(
                f"epoch = {iter}, "
                + "".ljust(13 - len(str(f"iter = {iter}, ")))
                + f"loss_u* = {loss_train_u_star[iter + total_iter_u_star]:.2e}, "
                f"mu_u* = {mu_u:.1e}\n"
                + "".ljust(13)
                + f"loss_v* = {loss_train_v_star[iter + total_iter_v_star]:.2e}, "
                f"mu_v* = {mu_v:.1e}"
            )
        elif u_star_not_done is True and v_star_not_done is not True:
            print(
                f"epoch = {epoch}, "
                f"loss_u* = {loss_train_u_star[epoch + total_epoch_u_star]:.2e}, "
                f"mu = {mu_u_star:.1e}"
            )
        elif u_star_not_done is not True and v_star_not_done is True:
            print(
                f"epoch = {epoch}, "
                f"loss_v* = {loss_train_v_star[epoch + total_epoch_v_star]:.2e}, "
                f"mu = {mu_v_star:.1e}"
            )
        else:
            print('Successful training ...')
            break

    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(savedloss_u, "k-", label="training loss")
    ax.semilogy(savedloss_u_valid, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.semilogy(savedloss_v, "k-", label="training loss")
    ax.semilogy(savedloss_v_valid, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

    return u_star_params, v_star_params


def main():
    # Define the neural network
    model = PinnModel([2, 20, 20, 1]).to(device)
    params = dict(model.named_parameters())
    print(model)

    total_params = model.num_total_params()
    print(f"Total number of parameters: {total_params}")

    # Define the training data
    mesh = CreateMesh()
    x_inner, y_inner = mesh.inner_points(100)
    x_bd, y_bd = mesh.boundary_points(100)
    x_inner_valid, y_inner_valid = mesh.inner_points(10000)
    x_bd_valid, y_bd_valid = mesh.boundary_points(1000)
    # Compute the boundary normal vector
    nx, ny = mesh.normal_vector(x_bd, y_bd)

    # Move the data to the device
    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
    x_inner_valid, y_inner_valid = x_inner_valid.to(device), y_inner_valid.to(device)
    x_bd_valid, y_bd_valid = x_bd_valid.to(device), y_bd_valid.to(device)
    nx, ny = nx.to(device), ny.to(device)

    # Pack the training data
    training_data = (
        (x_inner, y_inner),
        (x_bd, y_bd),
        (x_inner_valid, y_inner_valid),
        (x_bd_valid, y_bd_valid),
        (nx, ny),
    )

    u = functional_call(model, params, (x_inner, y_inner))
    print(f"Predicted u: {u.size()}")

    # Initialize the previous value
    prev_value = dict()
    prev_value["u0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "u")
    prev_value["v0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "v")
    prev_value["p0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "p")
    prev_value["u1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "u")
    prev_value["v1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "v")
    prev_value["p1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "p")
    prev_value["du0dx"], prev_value["du0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "u")
    prev_value["dv0dx"], prev_value["dv0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "v")
    prev_value["du1dx"], prev_value["du1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "u")
    prev_value["dv1dx"], prev_value["dv1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "v")
    prev_value["dp0dx"], prev_value["dp0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "p")
    prev_value["dp1dx"], prev_value["dp1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "p")

    prev_value_valid = dict()
    prev_value_valid["u0"] = exact_sol(x_inner_valid, y_inner_valid, 0.0 * Dt, Re, "u")
    prev_value_valid["v0"] = exact_sol(x_inner_valid, y_inner_valid, 0.0 * Dt, Re, "v")
    prev_value_valid["p0"] = exact_sol(x_inner_valid, y_inner_valid, 0.0 * Dt, Re, "p")
    prev_value_valid["u1"] = exact_sol(x_inner_valid, y_inner_valid, 1.0 * Dt, Re, "u")
    prev_value_valid["v1"] = exact_sol(x_inner_valid, y_inner_valid, 1.0 * Dt, Re, "v")
    prev_value_valid["p1"] = exact_sol(x_inner_valid, y_inner_valid, 1.0 * Dt, Re, "p")
    prev_value_valid["du0dx"], prev_value_valid["du0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 0.0 * Dt, Re, "u")
    prev_value_valid["dv0dx"], prev_value_valid["dv0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 0.0 * Dt, Re, "v")
    prev_value_valid["du1dx"], prev_value_valid["du1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 1.0 * Dt, Re, "u")
    prev_value_valid["dv1dx"], prev_value_valid["dv1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 1.0 * Dt, Re, "v")
    prev_value_valid["dp0dx"], prev_value_valid["dp0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 0.0 * Dt, Re, "p")
    prev_value_valid["dp1dx"], prev_value_valid["dp1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_valid.reshape(-1), y_inner_valid.reshape(-1), 1.0 * Dt, Re, "p")

    # reshape the values to (n, 1)
    for key, key_v in iter(zip(prev_value, prev_value_valid)):
        prev_value[key] = prev_value[key].reshape(-1, 1)
        prev_value_valid[key_v] = prev_value_valid[key_v].reshape(-1, 1)
        # print(f'prev_value["{key}"]: {prev_value[key].size()}')
        # print(f'prev_value_valid["{key_v}"]: {prev_value_valid[key_v].size()}')
    
    # Predict the intermediate velocity field (u*, v*)
    u_star_params, v_star_params = prediction_step(
        model, training_data, prev_value, prev_value_valid, 2
    )

    # Plot the predicted velocity field
    x_plot, y_plot = torch.meshgrid(
        torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing="xy"
    )
    x_plot, y_plot = x_plot.reshape(-1, 1).to(device), y_plot.reshape(-1, 1).to(device)
    u_star = model.predict(model, u_star_params, x_plot, y_plot).cpu().detach().numpy()
    v_star = model.predict(model, v_star_params, x_plot, y_plot).cpu().detach().numpy()

    fig, axs = plt.subplots(row=1, col=2, subplot_kw={"projection": "3d"})
    sca1 = axs[0].scatter(x_plot.cpu(), y_plot.cpu(), u_star, c=u_star, cmap="viridis")
    sca2 = axs[1].scatter(x_plot.cpu(), y_plot.cpu(), v_star, c=v_star, cmap="viridis")
    fig.colorbar(sca1, ax=axs[0])
    fig.colorbar(sca2, ax=axs[1])
    axs[0].set_xlabel("x")
    axs[1].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    Re = 400
    Dt = 0.01
    main()