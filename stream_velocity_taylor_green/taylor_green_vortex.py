"""
Taylor-Green Vortex flow using Physics-Informed Neural Networks (PINNs)
Stream function version

Incompressible Navier-Stokes Equations:
u_tt + u*u_x - v*u_y = - p_x - RE*Δu
v_tt + u*v_x + v*v_y = - p_y - RE*Δv
u_x + v_y = 0

Stream function:
ψ_y = u, ψ_x = -v

Projection Method:
Step 1: Predict the intermediate velocity field (u*, v*) using the neural network
Step 2: Project the intermediate velocity field onto the space of divergence-free fields
Step 3: Update the velocity and pressure fields
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad
from mesh_generator import CreateMesh
from utilities import exact_sol, qr_decomposition, cholesky


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

    def weights_init(self, model, *, multi=1.0):
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight.data, gain=multi)
            # nn.init.xavier_normal_(model.weight.data, gain=multi)

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
    
    def predict_dxy(self, model, params, x, y):
        """Compute the mixed directional derivative of the model output with respect to x and y."""
        output, vjpfunc = vjp(lambda primal: self.predict_dx(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_dyx(self, model, params, x, y):
        """Compute the mixed directional derivative of the model output with respect to y and x."""
        output, vjpfunc = vjp(lambda primal: self.predict_dy(model, params, primal, y), x)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dyy(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: self.predict_dy(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    

def prediction_step(model_u_star, model_v_star, training_data, prev_val, prev_val_valid, step):
    """The prediction step of the projection method"""
    # Unpack the training data
    x_inner, y_inner = training_data[0]
    x_bd, y_bd = training_data[1]
    x_inner_v, y_inner_v = training_data[2]
    x_bd_v, y_bd_v = training_data[3]

    # Move the data to the device
    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
    x_inner_v, y_inner_v = x_inner_v.to(device), y_inner_v.to(device)
    x_bd_v, y_bd_v = x_bd_v.to(device), y_bd_v.to(device)

    # Define the parameters
    u_star_params = dict(model_u_star.named_parameters())
    v_star_params = dict(model_v_star.named_parameters())
    # 5 times the parameters of phi_params
    model_u_star.weights_init(model_u_star, multi=5.0)
    model_v_star.weights_init(model_v_star, multi=5.0)
    # u_star_params_flatten = 5 * nn.utils.parameters_to_vector(u_star_params.values())
    # v_star_params_flatten = 5 * nn.utils.parameters_to_vector(v_star_params.values())
    # nn.utils.vector_to_parameters(u_star_params_flatten, u_star_params.values())
    # nn.utils.vector_to_parameters(v_star_params_flatten, v_star_params.values())

    # Compute the right-hand side values
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
    Rf_u_inner_valid = (
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
    Rf_v_inner_valid = (
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
    tol = 1.0e-9
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
            )(model_u_star, u_star_params, x_inner, y_inner, Rf_u_inner)

            jac_bd_dict_u = vmap(
                jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model_u_star, u_star_params, x_bd, y_bd, Rf_u_bd)

            # Stack the jacobian matrix
            jac_res_u = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict_u.values()])
            jac_bd_u = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict_u.values()])
            jac_res_u *= torch.sqrt(alpha_u / torch.tensor(x_inner.size(0)))
            jac_bd_u *= torch.sqrt(beta_u / torch.tensor(x_bd.size(0)))

            # Compute the residual of the loss function
            l_vec_res_u = compute_loss_res(model_u_star, u_star_params, x_inner, y_inner, Rf_u_inner)
            l_vec_bd_u = compute_loss_bd(model_u_star, u_star_params, x_bd, y_bd, Rf_u_bd)
            l_vec_res_u_valid = compute_loss_res(model_u_star, u_star_params, x_inner_v, y_inner_v, Rf_u_inner_valid)
            l_vec_bd_u_valid = compute_loss_bd(model_u_star, u_star_params, x_bd_v, y_bd_v, Rf_u_bd_valid)
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

            # Stop the training if the loss function is converged
            if (iter == Niter - 1) or (loss_u < tol):
                print('Successful training u* ...')
                u_star_finished = True

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
                        compute_loss_res(model_u_star, primal, x_inner, y_inner, Rf_u_inner) ** 2
                    ),
                    argnums=0,
                )(u_star_params)

                dloss_bd_dp_u = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd(model_u_star, primal, x_bd, y_bd, Rf_u_bd) ** 2
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
            )(model_v_star, v_star_params, x_inner, y_inner, Rf_v_inner)

            jac_bd_dict_v = vmap(
                jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, None, 0, 0, 0), out_dims=0
            )(model_v_star, v_star_params, x_bd, y_bd, Rf_v_bd)

            # Stack the jacobian matrix
            jac_res_v = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict_v.values()])
            jac_bd_v = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict_v.values()])
            jac_res_v *= torch.sqrt(alpha_v / torch.tensor(x_inner.size(0)))
            jac_bd_v *= torch.sqrt(beta_v / torch.tensor(x_bd.size(0)))

            # Compute the residual of the loss function
            l_vec_res_v = compute_loss_res(model_v_star, v_star_params, x_inner, y_inner, Rf_v_inner)
            l_vec_bd_v = compute_loss_bd(model_v_star, v_star_params, x_bd, y_bd, Rf_v_bd)
            l_vec_res_v_valid = compute_loss_res(model_v_star, v_star_params, x_inner_v, y_inner_v, Rf_v_inner_valid)
            l_vec_bd_v_valid = compute_loss_bd(model_v_star, v_star_params, x_bd_v, y_bd_v, Rf_v_bd_valid)
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

            # Stop the training if the loss function is converged
            if (iter == Niter - 1) or (loss_v < tol):
                print('Successful training v* ...')
                v_star_finished = True

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
                        compute_loss_res(model_v_star, primal, x_inner, y_inner, Rf_v_inner) ** 2
                    ),
                    argnums=0,
                )(v_star_params)

                dloss_bd_dp_v = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd(model_v_star, primal, x_bd, y_bd, Rf_v_bd) ** 2
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

        if u_star_finished is not True and v_star_finished is not True and iter % 5 == 0:
            print(
                f"iter = {iter}, "
                + "".ljust(13 - len(str(f"iter = {iter}, ")))
                + f"loss_u* = {loss_u:.2e}, "
                f"mu_u* = {mu_u:.1e}\n" + "".ljust(13) + f"loss_v* = {loss_v:.2e}, "
                f"mu_v* = {mu_v:.1e}"
            )
        elif u_star_finished is not True and v_star_finished is True and iter % 5 == 0:
            print(
                f"iter = {iter}, "
                f"loss_u* = {loss_u:.2e}, "
                f"mu = {mu_u:.1e}"
            )
        elif u_star_finished is True and v_star_finished is not True and iter % 5 == 0:
            print(
                f"iter = {iter}, "
                f"loss_v* = {loss_v:.2e}, "
                f"mu = {mu_v:.1e}"
            )
        if u_star_finished is True and v_star_finished is True:
            print('Successful training ...')
            print(
                f"iter = {iter}, "
                + "".ljust(13 - len(str(f"iter = {iter}, ")))
                + f"loss_u* = {loss_u:.2e}, "
                f"mu_u* = {mu_u:.1e}\n" + "".ljust(13) + f"loss_v* = {loss_v:.2e}, "
                f"mu_v* = {mu_v:.1e}"
            )
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


def projection_step(u_star_model, v_star_model, phi_model, psi_model, u_star_params, v_star_params, step):
    """The projection step of the projection method"""

    # Create the validation data
    mesh = CreateMesh()
    x_inner_v, y_inner_v = mesh.inner_points(10000)
    x_bd_v, y_bd_v = mesh.boundary_points(100)
    nx_v, ny_v = mesh.normal_vector(x_bd_v, y_bd_v)
    # Move the data to the device
    x_inner_v, y_inner_v = x_inner_v.to(device), y_inner_v.to(device)
    x_bd_v, y_bd_v = x_bd_v.to(device), y_bd_v.to(device)
    nx_v, ny_v = nx_v.to(device), ny_v.to(device)

    # Define the parameters
    phi_model.weights_init(phi_model, multi=10.0)
    psi_model.weights_init(psi_model, multi=10.0)
    phi_params = dict(phi_model.named_parameters())
    psi_params = dict(psi_model.named_parameters())
    num_params_phi = phi_model.num_total_params()
    num_params_psi = psi_model.num_total_params()
    
    # Compute the validate right-hand side values
    Rf_inner_1_valid = u_star_model.predict(u_star_model, u_star_params, x_inner_v, y_inner_v)
    Rf_inner_2_valid = v_star_model.predict(v_star_model, v_star_params, x_inner_v, y_inner_v)
    Rf_bd_1_valid = torch.zeros_like(x_bd_v)
    Rf_bd_2_valid = (
        exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "u") * nx_v +
        exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "v") * ny_v
    )

    def compute_loss_res_1(model1, model2, params1, params2, x, y, Rf_inner_1):
        """Compute the residual loss function."""
        pred = (
            model2.predict_dy(model2, params2, x, y) +
            (2 * Dt / 3) * model1.predict_dx(model1, params1, x, y)
        )
        loss_res = pred - Rf_inner_1
        return loss_res
    
    def compute_loss_res_2(model1, model2, params1, params2, x, y, Rf_inner_2):
        """Compute the residual loss function."""
        pred = (
            -model1.predict_dx(model2, params2, x, y) +
            (2 * Dt / 3) * model2.predict_dy(model1, params1, x, y)
        )
        loss_res = pred - Rf_inner_2
        return loss_res
    
    def compute_loss_bd_1(model1, model2, params1, params2, x, y, nx, ny, Rf_bd_1):
        """Compute the boundary loss function."""
        pred = (
            model1.predict_dx(model1, params1, x, y) * nx +
            model1.predict_dy(model1, params1, x, y) * ny
        )
        loss_bd = pred - Rf_bd_1
        return loss_bd
    
    def compute_loss_bd_2(model1, model2, params1, params2, x, y, nx, ny, Rf_bd_2):
        """Compute the boundary loss function."""
        pred = (
            model2.predict_dy(model2, params2, x, y) * nx -
            model2.predict_dx(model2, params2, x, y) * ny
        )
        loss_bd = pred - Rf_bd_2
        return loss_bd
    
    # Start training
    Niter = 1000
    tol = 1.0e-9
    alpha = 1.0
    beta = 1.0
    savedloss_phi = []
    savedloss_phi_valid = []
    overfitting = True

    while overfitting:
        mu = 1.0e3
        # Create the new trianing data
        x_inner, y_inner = mesh.inner_points(100)
        x_bd, y_bd = mesh.boundary_points(10)
        nx, ny = mesh.normal_vector(x_bd, y_bd)

        # Move the data to the device
        x_inner, y_inner = x_inner.to(device), y_inner.to(device)
        x_bd, y_bd = x_bd.to(device), y_bd.to(device)
        nx, ny = nx.to(device), ny.to(device)

        # Compute the right-hand side values
        Rf_inner_1 = u_star_model.predict(u_star_model, u_star_params, x_inner, y_inner)
        Rf_inner_2 = v_star_model.predict(v_star_model, v_star_params, x_inner, y_inner)
        Rf_bd_1 = torch.zeros_like(x_bd)
        Rf_bd_2 = (
            exact_sol(x_bd, y_bd, step * Dt, Re, "u") * nx +
            exact_sol(x_bd, y_bd, step * Dt, Re, "v") * ny
        )

        for iter in range(Niter):
            # Compute the jacobi matrix
            jac_res_1_phi_dict, jac_res_1_psi_dict = vmap(
                jacrev(compute_loss_res_1, argnums=(2, 3)),
                in_dims=(None, None, None, None, 0, 0, 0),
                out_dims=0,
            )(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_inner_1)

            jac_res_2_phi_dict, jac_res_2_psi_dict = vmap(
                jacrev(compute_loss_res_2, argnums=(2, 3)),
                in_dims=(None, None, None, None, 0, 0, 0),
                out_dims=0,
            )(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_inner_2)

            jac_bd_1_phi_dict, jac_bd_1_psi_dict = vmap(
                jacrev(compute_loss_bd_1, argnums=(2, 3)),
                in_dims=(None, None, None, None, 0, 0, 0, 0, 0),
                out_dims=0,
            )(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, nx, ny, Rf_bd_1)

            jac_bd_2_phi_dict, jac_bd_2_psi_dict = vmap(
                jacrev(compute_loss_bd_2, argnums=(2, 3)),
                in_dims=(None, None, None, None, 0, 0, 0, 0, 0),
                out_dims=0,
            )(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, nx, ny, Rf_bd_2)

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
            # print(f"jac_res_1_phi: {jac_res_1_phi.size()}")
            # print(f"jac_res_1_psi: {jac_res_1_psi.size()}")
            # print(f'jac_bd_1_phi: {jac_bd_1_phi.size()}')
            # print(f'jac_bd_1_psi: {jac_bd_1_psi.size()}')

            # Compute the residual of the loss function
            l_vec_res_1 = compute_loss_res_1(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_inner_1)
            l_vec_res_2 = compute_loss_res_2(phi_model, psi_model, phi_params, psi_params, x_inner, y_inner, Rf_inner_2)
            l_vec_bd_1 = compute_loss_bd_1(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, nx, ny, Rf_bd_1)
            l_vec_bd_2 = compute_loss_bd_2(phi_model, psi_model, phi_params, psi_params, x_bd, y_bd, nx, ny, Rf_bd_2)
            l_vec_res_1_valid = compute_loss_res_1(phi_model, psi_model, phi_params, psi_params, x_inner_v, y_inner_v, Rf_inner_1_valid)
            l_vec_res_2_valid = compute_loss_res_2(phi_model, psi_model, phi_params, psi_params, x_inner_v, y_inner_v, Rf_inner_2_valid)
            l_vec_bd_1_valid = compute_loss_bd_1(phi_model, psi_model, phi_params, psi_params, x_bd_v, y_bd_v, nx_v, ny_v, Rf_bd_1_valid)
            l_vec_bd_2_valid = compute_loss_bd_2(phi_model, psi_model, phi_params, psi_params, x_bd_v, y_bd_v, nx_v, ny_v, Rf_bd_2_valid)
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
            # print(f'jacobian: {jacobian.size()}, l_vec: {l_vec.size()}')

            # Solve the non-linear system
            # p = cholesky(jacobian, l_vec, mu, device)
            p = qr_decomposition(jacobian, l_vec, mu, device)
            # print(f'p: {p.size()}')
            # Update the parameters
            phi_params_flatten = nn.utils.parameters_to_vector(phi_params.values())
            psi_params_flatten = nn.utils.parameters_to_vector(psi_params.values())
            phi_params_flatten += p[:num_params_phi]
            psi_params_flatten += p[num_params_phi:]
            nn.utils.vector_to_parameters(phi_params_flatten, phi_params.values())
            nn.utils.vector_to_parameters(psi_params_flatten, psi_params.values())

            # Compute the loss function
            loss_phi = (
                torch.sum(l_vec_res_1**2 + l_vec_res_2**2) + 
                torch.sum(l_vec_bd_1**2 + l_vec_bd_2**2)
            )
            loss_phi_valid = (
                torch.sum(l_vec_res_1_valid**2 + l_vec_res_2_valid**2) + 
                torch.sum(l_vec_bd_1_valid**2 + l_vec_bd_2_valid**2)
            )
            savedloss_phi.append(loss_phi.item())
            savedloss_phi_valid.append(loss_phi_valid.item())

            # Update alpha and beta
            if iter % 100 == 0:
                dloss_res_1_dp = grad(
                    lambda primal: torch.sum(
                        compute_loss_res_1(phi_model, psi_model, primal, psi_params, x_inner, y_inner, Rf_inner_1) ** 2
                    ),
                    argnums=0,
                )(phi_params)

                dloss_res_2_dp = grad(
                    lambda primal: torch.sum(
                        compute_loss_res_2(phi_model, psi_model, phi_params, primal, x_inner, y_inner, Rf_inner_2) ** 2
                    ),
                    argnums=0,
                )(phi_params)

                dloss_bd_1_dp = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd_1(phi_model, psi_model, primal, psi_params, x_bd, y_bd, nx, ny, Rf_bd_1) ** 2
                    ),
                    argnums=0,
                )(phi_params)

                dloss_bd_2_dp = grad(
                    lambda primal: torch.sum(
                        compute_loss_bd_2(phi_model, psi_model, phi_params, primal, x_bd, y_bd, nx, ny, Rf_bd_2) ** 2
                    ),
                    argnums=0,
                )(phi_params)

                dloss_res_1_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_res_1_dp.values()
                ) / torch.tensor(2 * x_inner.size(0))
                dloss_res_2_dp_flatten = nn.utils.parameters_to_vector(
                    dloss_res_2_dp.values()
                ) / torch.tensor(2 * x_inner.size(0))
                dloss_res_dp_flatten = torch.hstack((dloss_res_1_dp_flatten, dloss_res_2_dp_flatten))
                dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)

                dloss_bd_dp_1_flatten = nn.utils.parameters_to_vector(
                    dloss_bd_1_dp.values()
                ) / torch.tensor(2 * x_bd.size(0))
                dloss_bd_dp_2_flatten = nn.utils.parameters_to_vector(
                    dloss_bd_2_dp.values()
                ) / torch.tensor(2 * x_bd.size(0))
                dloss_bd_dp_flatten = torch.hstack((dloss_bd_dp_1_flatten, dloss_bd_dp_2_flatten))
                dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)

                alpha_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_res_dp_norm
                beta_bar = (dloss_res_dp_norm + dloss_bd_dp_norm) / dloss_bd_dp_norm

                alpha = (1 - 0.1) * alpha + 0.1 * alpha_bar
                beta = (1 - 0.1) * beta + 0.1 * beta_bar
                print(f"alpha: {alpha:.2f}, beta: {beta:.2f}")

            if iter % 5 == 0:
                print(f"iter = {iter}, loss_phi = {loss_phi.item():.2e}, mu = {mu:.1e}")

            # Stop the training if the loss function is converged
            if (iter == Niter - 1) or (loss_phi < tol):
                print(f"iter = {iter}, loss_phi = {loss_phi.item():.2e}, mu = {mu:.1e}")
                if (loss_phi_valid / loss_phi) < 10.0: 
                    overfitting = False
                    print('Successful training phi ...')
                else:
                    print('Overfitting ...')
                    overfitting = False
                break
                
            # Update the parameter mu
            if iter % 3 == 0:
                if savedloss_phi[iter] > savedloss_phi[iter - 1]:
                    mu = min(2 * mu, 1e8)
                else:
                    mu = max(mu / 3, 1e-10)

        # Plot the loss function
        fig, ax = plt.subplots()
        ax.semilogy(savedloss_phi, "k-", label="training loss")
        ax.semilogy(savedloss_phi_valid, "r--", label="test loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.show()

    return phi_params, psi_params


def update_step(training_data, u_star_model, v_star_model, phi_model, 
                u_star_params, v_star_params, phi_params, prev_value, 
                prev_value_valid):
    """The update step for velocity and pressure fields"""
    # Unpack the training data
    x_inner, y_inner = training_data[0]
    x_bd, y_bd = training_data[1]
    x_inner_v, y_inner_v = training_data[2]
    x_bd_v, y_bd_v = training_data[3]

    x_training = torch.vstack((x_inner, x_bd))
    y_training = torch.vstack((y_inner, y_bd))
    x_test = torch.vstack((x_inner_v, x_bd_v))
    y_test = torch.vstack((y_inner_v, y_bd_v))

    # Move the data to the device
    x_training, y_training = x_training.to(device), y_training.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Create the new dictionary for the update value
    new_value = dict()
    new_value_valid = dict()
    new_value["u0"] = prev_value["u1"]
    new_value["v0"] = prev_value["v1"]
    new_value["p0"] = prev_value["p1"]
    new_value["du0dx"] = prev_value["du1dx"]
    new_value["du0dy"] = prev_value["du1dy"]
    new_value["dv0dx"] = prev_value["dv1dx"]
    new_value["dv0dy"] = prev_value["dv1dy"]
    new_value["dp0dx"] = prev_value["dp1dx"]
    new_value["dp0dy"] = prev_value["dp1dy"]

    new_value_valid["u0"] = prev_value_valid["u1"]
    new_value_valid["v0"] = prev_value_valid["v1"]
    new_value_valid["p0"] = prev_value_valid["p1"]
    new_value_valid["du0dx"] = prev_value_valid["du1dx"]
    new_value_valid["du0dy"] = prev_value_valid["du1dy"]
    new_value_valid["dv0dx"] = prev_value_valid["dv1dx"]
    new_value_valid["dv0dy"] = prev_value_valid["dv1dy"]
    new_value_valid["dp0dx"] = prev_value_valid["dp1dx"]
    new_value_valid["dp0dy"] = prev_value_valid["dp1dy"]

    new_value["u1"] = (
        u_star_model.predict(u_star_model, u_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dx(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["v1"] = (
        v_star_model.predict(v_star_model, v_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dy(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["p1"] = (
        torch.tensor(prev_value["p0"], device=device)
        + phi_model.predict(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dx(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    new_value["du1dx"] = (
        u_star_model.predict_dx(u_star_model, u_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dxx(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["du1dy"] = (
        u_star_model.predict_dy(u_star_model, u_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dxy(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["dv1dx"] = (
        v_star_model.predict_dx(v_star_model, v_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dyx(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["dv1dy"] = (
        v_star_model.predict_dy(v_star_model, v_star_params, x_training, y_training)
        - (2 * Dt / 3) * phi_model.predict_dyy(phi_model, phi_params, x_training, y_training)
    ).cpu().detach().numpy()
    new_value["dp1dx"] = (
        torch.tensor(prev_value["dp0dx"], device=device)
        + phi_model.predict_dx(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dxx(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dyx(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    new_value["dp1dy"] = (
        torch.tensor(prev_value["dp0dy"], device=device)
        + phi_model.predict_dy(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dxy(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dyy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()

    new_value_valid["u1"] = (
        u_star_model.predict(u_star_model, u_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dx(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["v1"] = (
        v_star_model.predict(v_star_model, v_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dy(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["p1"] = (
        torch.tensor(prev_value_valid["p0"], device=device)
        + phi_model.predict(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dx(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    new_value_valid["du1dx"] = (
        u_star_model.predict_dx(u_star_model, u_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dxx(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["du1dy"] = (
        u_star_model.predict_dy(u_star_model, u_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dxy(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["dv1dx"] = (
        v_star_model.predict_dx(v_star_model, v_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dyx(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["dv1dy"] = (
        v_star_model.predict_dy(v_star_model, v_star_params, x_test, y_test)
        - (2 * Dt / 3) * phi_model.predict_dyy(phi_model, phi_params, x_test, y_test)
    ).cpu().detach().numpy()
    new_value_valid["dp1dx"] = (
        torch.tensor(prev_value_valid["dp0dx"], device=device)
        + phi_model.predict_dx(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dxx(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dyx(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    new_value_valid["dp1dy"] = (
        torch.tensor(prev_value_valid["dp0dy"], device=device)
        + phi_model.predict_dy(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dxy(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dyy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()

    return new_value, new_value_valid


def main():
    # Define the neural network
    u_star_model = PinnModel([2, 40, 1]).to(device)
    v_star_model = PinnModel([2, 40, 1]).to(device)
    phi_model = PinnModel([2, 100, 1]).to(device)
    psi_model = PinnModel([2, 100, 1]).to(device)

    total_params_u_star = u_star_model.num_total_params()
    total_params_v_star = v_star_model.num_total_params()
    total_params_phi = phi_model.num_total_params()
    total_params_psi = psi_model.num_total_params()
    print(f"Total number of u* parameters: {total_params_u_star}")
    print(f"Total number of v* parameters: {total_params_v_star}")
    print(f"Total number of phi parameters: {total_params_phi}")
    print(f"Total number of psi parameters: {total_params_psi}")

    # Define the training data
    mesh = CreateMesh()
    x_inner, y_inner = mesh.inner_points(100)
    x_bd, y_bd = mesh.boundary_points(10)
    x_inner_valid, y_inner_valid = mesh.inner_points(90000)
    x_bd_valid, y_bd_valid = mesh.boundary_points(300)
    # Compute the boundary normal vector

    # Pack the training data
    training_data = (
        (x_inner, y_inner),
        (x_bd, y_bd),
        (x_inner_valid, y_inner_valid),
        (x_bd_valid, y_bd_valid),
    )

    # Initialize the previous value
    x_training = torch.vstack((x_inner, x_bd))
    y_training = torch.vstack((y_inner, y_bd))
    x_test = torch.vstack((x_inner_valid, x_bd_valid))
    y_test = torch.vstack((y_inner_valid, y_bd_valid))
    
    prev_value = dict()
    prev_value["x_data"] = x_training
    prev_value["y_data"] = y_training
    prev_value["u0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["v0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["p0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["u1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["v1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["p1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "p").detach().numpy()
    prev_value["du0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["du0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["dv0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["dv0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["du1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["du1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["dv1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["dv1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["dp0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()

    prev_value_valid = dict()
    prev_value_valid["x_data"] = x_test
    prev_value_valid["y_data"] = y_test
    prev_value_valid["u0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["v0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["p0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["u1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["v1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["p1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["du0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["du0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["dv0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dv0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["du1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["du1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["dv1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dv1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dp0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()

    # reshape the values to (n, 1)
    for key, key_v in iter(zip(prev_value, prev_value_valid)):
        prev_value[key] = prev_value[key].reshape(-1, 1)
        prev_value_valid[key_v] = prev_value_valid[key_v].reshape(-1, 1)
        # print(f'prev_value["{key}"]: {prev_value[key].size()}')
        # print(f'prev_value_valid["{key_v}"]: {prev_value_valid[key_v].size()}')
    
    # Predict the intermediate velocity field (u*, v*)
    u_star_params, v_star_params = prediction_step(
        u_star_model, v_star_model, training_data, prev_value, prev_value_valid, 2
    )
    u_star_model.load_state_dict(u_star_params)
    v_star_model.load_state_dict(v_star_params)
    print("Finish the prediction step ...")

    # Project the intermediate velocity field onto the space of divergence-free fields
    phi_params, psi_params = projection_step(
        u_star_model, v_star_model, phi_model, psi_model, u_star_params, v_star_params, 2
    )
    phi_model.load_state_dict(phi_params)
    psi_model.load_state_dict(psi_params)
    print("Finish the projection step ...")

    # Update the velocity field and pressure field
    new_value, new_value_valid = update_step(
        training_data,
        u_star_model,
        v_star_model,
        phi_model,
        u_star_params,
        v_star_params,
        phi_params,
        prev_value,
        prev_value_valid
    )

    prev_value.update(new_value)
    prev_value_valid.update(new_value_valid)
    print("Finish the update step ...")

    # Plot the intermediate velocity field and phi field
    x_plot = torch.vstack((x_inner_valid, x_bd_valid))
    y_plot = torch.vstack((y_inner_valid, y_bd_valid))
    x_plot, y_plot = x_plot.reshape(-1, 1), y_plot.reshape(-1, 1)
    u_star = u_star_model.predict(u_star_model, u_star_params, x_plot.to(device), y_plot.to(device)).cpu().detach().numpy()
    v_star = v_star_model.predict(v_star_model, v_star_params, x_plot.to(device), y_plot.to(device)).cpu().detach().numpy()
    phi = phi_model.predict(phi_model, phi_params, x_plot.to(device), y_plot.to(device)).cpu().detach().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"})
    sca1 = axs[0].scatter(x_plot, y_plot, u_star, c=u_star, cmap="jet", s=1)
    sca2 = axs[1].scatter(x_plot, y_plot, v_star, c=v_star, cmap="jet", s=1)
    sca3 = axs[2].scatter(x_plot, y_plot, phi, c=phi, cmap="jet", s=1)
    fig.colorbar(sca1, ax=axs[0], shrink=0.3, aspect=10, pad=0.15)
    fig.colorbar(sca2, ax=axs[1], shrink=0.3, aspect=10, pad=0.15)
    fig.colorbar(sca3, ax=axs[2], shrink=0.3, aspect=10, pad=0.15)
    axs[0].set_xlabel("x")
    axs[1].set_xlabel("x")
    axs[2].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].set_ylabel("y")
    axs[2].set_ylabel("y")
    axs[0].set_title("Predicted u*")
    axs[1].set_title("Predicted v*")
    axs[2].set_title("Predicted phi")
    plt.show()

    # Plot the velocity field and pressure field
    u_pred = prev_value_valid["u1"]
    v_pred = prev_value_valid["v1"]
    p_pred = prev_value_valid["p1"]
    fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"})
    sca1 = axs[0].scatter(x_plot, y_plot, u_pred, c=u_pred, cmap="jet", s=1)
    sca2 = axs[1].scatter(x_plot, y_plot, v_pred, c=v_pred, cmap="jet", s=1)
    sca3 = axs[2].scatter(x_plot, y_plot, p_pred, c=p_pred, cmap="jet", s=1)
    fig.colorbar(sca1, ax=axs[0], shrink=0.3, aspect=10, pad=0.15)
    fig.colorbar(sca2, ax=axs[1], shrink=0.3, aspect=10, pad=0.15)
    fig.colorbar(sca3, ax=axs[2], shrink=0.3, aspect=10, pad=0.15)
    axs[0].set_xlabel("x")
    axs[1].set_xlabel("x")
    axs[2].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].set_ylabel("y")
    axs[2].set_ylabel("y")
    axs[0].set_title("Predicted u")
    axs[1].set_title("Predicted v")
    axs[2].set_title("Predicted p")
    plt.show()


if __name__ == "__main__":
    Re = 400
    Dt = 0.01
    main()