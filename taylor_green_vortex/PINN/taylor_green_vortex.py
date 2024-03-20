"""
Taylor-Green Vortex

u_tt + u*u_x - v*u_y = - p_x - RE*Δu
v_tt + u*v_x + v*v_y = - p_y - RE*Δv
u_x + v_y = 0
"""

import os
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.func import jacrev, vmap, grad
from utilities import Model, MatrixSolver
from utilities import exact_sol, compute_u_star_rhs, compute_u_star_bdy_value
from mesh_generator import CreateMesh

# Check if the directory exists
pwd = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(pwd + "/data"):
    print("Creating data directory...")
    os.makedirs(pwd + "/data")
else:
    print("Data directory already exists...")

if not os.path.exists(pwd + "/figure"):
    print("Creating figure directory...")
    os.makedirs(pwd + "/figure")
else:
    print("Figure directory already exists...")

device_cpu = "cpu"
device_gpu = "cuda"
# Set the number of threads used in PyTorch
torch.set_num_threads(10)
print(f"Using {torch.get_num_threads()} threads in PyTorch")

# Set the default device to cpu
torch.set_default_device(device_cpu)
torch.set_default_dtype(torch.float64)


RE = 400
DT = 0.01


class Prediction:
    def __init__(self, model, solver, training_p, test_p, *, batch=1):
        self.model = model
        self.solver = solver
        self.batch = batch

        (
            self.training_domain,
            self.training_left_boundary,
            self.training_right_boundary,
            self.training_top_boundary,
            self.training_bottom_boundary,
        ) = training_p
        (
            self.test_domain,
            self.test_left_boundary,
            self.test_right_boundary,
            self.test_top_boundary,
            self.test_bottom_boundary,
        ) = test_p

        self.training_boundary = torch.vstack((
            self.training_left_boundary,
            self.training_right_boundary,
            self.training_top_boundary,
            self.training_bottom_boundary,
        ))
        self.test_boundary = torch.vstack((
            self.test_left_boundary,
            self.test_right_boundary,
            self.test_top_boundary,
            self.test_bottom_boundary,
        ))

        # Move the data to model device
        self.test_domain = self.test_domain.to(self.model.device)
        self.test_boundary = self.test_boundary.to(self.model.device)
        
    def main_eq(self, params, points_x, points_y):
        func = (
            + 3 * self.model.forward_2d(params, points_x, points_y)
            - (2 * DT / RE) * (
                self.model.forward_2d_dxx(params, points_x, points_y)
                +
                self.model.forward_2d_dyy(params, points_x, points_y)
            )
        )
        return func
    
    def boundary_eq(self, params, points_x, points_y):
        func = self.model.forward_2d(params, points_x, points_y)
        return func
    
    def jacobian_main_eq(self, params, points_x, points_y):
        jacobian = vmap(
            jacrev(self.main_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y)
        row_size = points_x.size(0)
        col_size = torch.sum(torch.tensor([torch.add(weight.numel(), bias.numel()) 
                                           for weight, bias in params]))
        jac = torch.zeros(row_size, col_size, device=self.model.device)
        loc = 0
        for idx, (jac_w, jac_b) in enumerate(jacobian):
            jac[:, loc:loc+params[idx][0].numel()] = jac_w.view(points_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[:, loc:loc+params[idx][1].numel()] = jac_b.view(points_x.size(0), -1)
            loc += params[idx][1].numel()
        return torch.squeeze(jac)
    
    def jacobian_boundary_eq(self, params, points_x, points_y):
        jacobian = vmap(
            jacrev(self.boundary_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y)
        row_size = points_x.size(0)
        col_size = torch.sum(torch.tensor([torch.add(weight.numel(), bias.numel()) 
                                           for weight, bias in params]))
        jac = torch.zeros(row_size, col_size, device=self.model.device)
        loc = 0
        for idx, (jac_w, jac_b) in enumerate(jacobian):
            jac[:, loc:loc+params[idx][0].numel()] = jac_w.view(points_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[:, loc:loc+params[idx][1].numel()] = jac_b.view(points_x.size(0), -1)
            loc += params[idx][1].numel()
        return torch.squeeze(jac)
    
    def main_eq_residual(self, params, points_x, points_y, target):
        diff = (
            target
            -
            vmap(self.main_eq, in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y)
        )
        return diff
    
    def boundary_eq_residual(self, params, points_x, points_y, target):
        diff = (
            target
            -
            vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y)
        )
        return diff
    
    def loss_main_eq(self, params, points_x, points_y, target):
        mse = torch.nn.MSELoss(reduction="mean")(
            vmap(self.main_eq, in_dims=(None, 0, 0), out_dims=0)(params, points_x, points_y),
            target
        )
        return mse
    
    def loss_boundary_eq(self, params, points_x, points_y, target):
        mse = torch.nn.MSELoss(reduction="mean")(
            vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(params, points_x, points_y),
            target
        )
        return mse

    def learning_process(self, params_u0, params_v0, params_p0, params_u1, params_v1, step):
        batch_domain = Data.DataLoader(
            dataset=Data.TensorDataset(self.training_domain[:, 0], self.training_domain[:, 1]),
            batch_size=int(self.training_domain[:, 0].size(0) / self.batch),
            shuffle=True,
            drop_last=True
        )
        batch_boundary = Data.DataLoader(
            dataset=Data.TensorDataset(self.training_boundary[:, 0], self.training_boundary[:, 1]),
            batch_size=int(self.training_boundary[:, 0].size(0) / self.batch),
            shuffle=True,
            drop_last=True
        )

        params_u_star = self.model.initialize_mlp(multi=1.0)
        params_v_star = self.model.initialize_mlp(multi=1.0)

        # Initialize the parameters
        Niter = 10**4
        tol = 1e-10
        total_epoch_u_star = 0
        total_epoch_v_star = 0
        u_star_not_done = True
        v_star_not_done = True
        # Initialize the loss function
        loss_train_u_star = torch.zeros(self.batch * Niter, device=self.model.device)
        loss_test_u_star = torch.zeros(self.batch * Niter, device=self.model.device)
        loss_train_v_star = torch.zeros(self.batch * Niter, device=self.model.device)
        loss_test_v_star = torch.zeros(self.batch * Niter, device=self.model.device)


        for batch_step, (data_domain, data_boundary) in enumerate(
            zip(batch_domain, batch_boundary)):
            print(f"Batch step = {batch_step}")

            # Move the data to GPU
            for idx in torch.arange(2):
                data_domain[idx] = data_domain[idx].to(self.model.device)
                data_boundary[idx] = data_boundary[idx].to(self.model.device)

            # Initialize the parameters
            mu_u_star = torch.tensor(10**3, device=self.model.device)
            mu_v_star = torch.tensor(10**3, device=self.model.device)
            alpha_u_star = torch.tensor(1.0, device=self.model.device)
            alpha_v_star = torch.tensor(1.0, device=self.model.device)
            beta_u_star = torch.tensor(1.0, device=self.model.device)
            beta_v_star = torch.tensor(1.0, device=self.model.device)

            # Compute the right hand side of the equation
            u_star_rhs, v_star_rhs = compute_u_star_rhs(self.model,
                params_u0, params_v0, params_p0, params_u1, params_v1, 
                data_domain[0], data_domain[1], step)
            u_star_rhs_test, v_star_rhs_test = compute_u_star_rhs(self.model,
                params_u0, params_v0, params_p0, params_u1, params_v1,
                self.test_domain[:, 0], self.test_domain[:, 1], step)
            u_star_bdy_rhs, v_star_bdy_rhs = compute_u_star_bdy_value(
                data_boundary[0], data_boundary[1], step * DT, RE)
            u_star_bdy_rhs_test, v_star_bdy_rhs_test = compute_u_star_bdy_value(
                self.test_boundary[:, 0], self.test_boundary[:, 1], step * DT, RE)

            # Start the training process
            for epoch in range(Niter):
                if u_star_not_done:
                    # Compute each jacobian matrix
                    jac_main_eq = self.jacobian_main_eq(
                        params_u_star, data_domain[0], data_domain[1])
                    jac_boundary_eq = self.jacobian_boundary_eq(
                        params_u_star, data_boundary[0], data_boundary[1])
                    # Combine the jacobian matrix
                    jacobian = torch.vstack(
                        (
                            jac_main_eq
                            * torch.sqrt(alpha_u_star / torch.tensor(data_domain[0].size(0))),
                            jac_boundary_eq
                            * torch.sqrt(beta_u_star / torch.tensor(data_boundary[0].size(0))),
                        )
                    )

                    # Compute each residual
                    residual_main_eq = self.main_eq_residual(
                        params_u_star, data_domain[0], data_domain[1], u_star_rhs)
                    residual_boundary_eq = self.boundary_eq_residual(
                        params_u_star, data_boundary[0], data_boundary[1], 
                        u_star_bdy_rhs)
                    # Combine the residual
                    residual = torch.hstack(
                        (
                            residual_main_eq
                            * torch.sqrt(alpha_u_star / torch.tensor(data_domain[0].size(0))),
                            residual_boundary_eq
                            * torch.sqrt(beta_u_star / torch.tensor(data_boundary[0].size(0))),
                        )
                    )

                    # Compute the update value of weight and bias
                    # h = self.solver.cholesky(jacobian, diff, mu)
                    h = self.solver.qr_decomposition(jacobian, residual.view(-1, 1), mu_u_star)
                    params_u_star_flatten = self.model.flatten_params(params_u_star)
                    params_u_star_flatten = torch.add(params_u_star_flatten, h)
                    params_u_star = self.model.unflatten_params(params_u_star_flatten)

                    # Compute the loss function
                    loss_train_u_star[epoch + total_epoch_u_star] = (
                        alpha_u_star * self.loss_main_eq(
                            params_u_star, data_domain[0], data_domain[1], u_star_rhs) 
                        + 
                        beta_u_star * self.loss_boundary_eq(
                            params_u_star, data_boundary[0], data_boundary[1], u_star_bdy_rhs
                        )
                    )

                    loss_test_u_star[epoch + total_epoch_u_star] = (
                        self.loss_main_eq(
                            params_u_star, self.test_domain[:, 0], 
                            self.test_domain[:, 1], u_star_rhs_test,
                        ) 
                        + 
                        self.loss_boundary_eq(
                            params_u_star, self.test_boundary[:, 0],
                            self.test_boundary[:, 1], u_star_bdy_rhs_test,
                        )
                    )
                    # Update alpha and beta
                    if epoch % 500 == 0:
                        residual_params_alpha = grad(self.loss_main_eq, argnums=0)(
                            params_u_star, data_domain[0], data_domain[1], u_star_rhs
                        )
                        loss_domain_deriv_params = torch.linalg.vector_norm(
                            self.model.flatten_params(residual_params_alpha)
                        )
                        residual_params_beta = grad(self.loss_boundary_eq, argnums=0)(
                            params_u_star, data_boundary[0], data_boundary[1], 
                            u_star_bdy_rhs
                        )
                        loss_boundary_deriv_params = torch.linalg.vector_norm(
                            self.model.flatten_params(residual_params_beta)
                        )

                        alpha_bar = (
                            loss_domain_deriv_params + loss_boundary_deriv_params
                        ) / loss_domain_deriv_params
                        beta_bar = (
                            loss_domain_deriv_params + loss_boundary_deriv_params
                        ) / loss_boundary_deriv_params

                        alpha_u_star = (1 - 0.1) * alpha_u_star + 0.1 * alpha_bar
                        beta_u_star = (1 - 0.1) * beta_u_star + 0.1 * beta_bar
                        print(f"alpha_u_star = {alpha_u_star:.3f}, beta_u_star = {beta_u_star:.3f}")

                    # Stop this batch if the loss function is small enough
                    if loss_train_u_star[epoch + total_epoch_u_star] < tol:
                        print('Successful training u star...')
                        total_epoch_u_star = total_epoch_u_star + epoch + 1
                        u_star_not_done = False

                    elif epoch % 5 == 0 and epoch > 0:
                        if (loss_train_u_star[epoch + total_epoch_u_star] > 
                            loss_train_u_star[epoch + total_epoch_u_star - 1]):
                            mu_u_star = torch.min(torch.tensor((2 * mu_u_star, 1e8)))

                        if (loss_train_u_star[epoch + total_epoch_u_star] <
                            loss_train_u_star[epoch + total_epoch_u_star - 1]):
                            mu_u_star = torch.max(torch.tensor((mu_u_star / 3, 1e-12)))

                    elif (
                        loss_train_u_star[epoch + total_epoch_u_star]
                        / loss_train_u_star[epoch + total_epoch_u_star - 1]
                        > 100
                    ) and epoch > 0:
                        mu_u_star = torch.min(torch.tensor((2 * mu_u_star, 1e8)))

                if v_star_not_done:
                    # Compute each jacobian matrix
                    jac_main_eq = self.jacobian_main_eq(
                        params_v_star, data_domain[0], data_domain[1])
                    jac_boundary_eq = self.jacobian_boundary_eq(
                        params_v_star, data_boundary[0], data_boundary[1])
                    # Combine the jacobian matrix
                    jacobian = torch.vstack(
                        (
                            jac_main_eq
                            * torch.sqrt(alpha_v_star / torch.tensor(data_domain[0].size(0))),
                            jac_boundary_eq
                            * torch.sqrt(beta_v_star / torch.tensor(data_boundary[0].size(0))),
                        )
                    )

                    # Compute each residual
                    residual_main_eq = self.main_eq_residual(
                        params_v_star, data_domain[0], data_domain[1], v_star_rhs)
                    residual_boundary_eq = self.boundary_eq_residual(
                        params_v_star, data_boundary[0], data_boundary[1], 
                        v_star_bdy_rhs)
                    # Combine the residual
                    residual = torch.hstack(
                        (
                            residual_main_eq
                            * torch.sqrt(alpha_v_star / torch.tensor(data_domain[0].size(0))),
                            residual_boundary_eq
                            * torch.sqrt(beta_v_star / torch.tensor(data_boundary[0].size(0))),
                        )
                    )

                    # Compute the update value of weight and bias
                    # h = self.solver.cholesky(jacobian, diff, mu)
                    h = self.solver.qr_decomposition(jacobian, residual.view(-1, 1), mu_v_star)
                    params_v_star_flatten = self.model.flatten_params(params_v_star)
                    params_v_star_flatten = torch.add(params_v_star_flatten, h)
                    params_v_star = self.model.unflatten_params(params_v_star_flatten)

                    # Compute the loss function
                    loss_train_v_star[epoch + total_epoch_v_star] = (
                        alpha_v_star * self.loss_main_eq(
                            params_v_star, data_domain[0], data_domain[1], v_star_rhs)
                        +
                        beta_v_star * self.loss_boundary_eq(
                            params_v_star, data_boundary[0], data_boundary[1], v_star_bdy_rhs
                        )
                    )

                    loss_test_v_star[epoch + total_epoch_v_star] = (
                        self.loss_main_eq(
                            params_v_star,
                            self.test_domain[:, 0],
                            self.test_domain[:, 1],
                            v_star_rhs_test,
                        )
                        +
                        self.loss_boundary_eq(
                            params_v_star,
                            self.test_boundary[:, 0],
                            self.test_boundary[:, 1],
                            v_star_bdy_rhs_test,
                        )
                    )

                    # Update alpha and beta
                    if epoch % 500 == 0:
                        residual_params_alpha = grad(self.loss_main_eq, argnums=0)(
                            params_v_star, data_domain[0], data_domain[1], v_star_rhs
                        )
                        loss_domain_deriv_params = torch.linalg.vector_norm(
                            self.model.flatten_params(residual_params_alpha)
                        )
                        residual_params_beta = grad(self.loss_boundary_eq, argnums=0)(
                            params_v_star, data_boundary[0], data_boundary[1], 
                            v_star_bdy_rhs
                        )
                        loss_boundary_deriv_params = torch.linalg.vector_norm(
                            self.model.flatten_params(residual_params_beta)
                        )

                        alpha_bar = (
                            loss_domain_deriv_params + loss_boundary_deriv_params
                        ) / loss_domain_deriv_params
                        beta_bar = (
                            loss_domain_deriv_params + loss_boundary_deriv_params
                        ) / loss_boundary_deriv_params

                        alpha_v_star = (1 - 0.1) * alpha_v_star + 0.1 * alpha_bar
                        beta_v_star = (1 - 0.1) * beta_v_star + 0.1 * beta_bar
                        print(f'alpha_v_star = {alpha_v_star:.3f}, beta_v_star = {beta_v_star:.3f}')

                    # Stop this batch if the loss function is small enough
                    if loss_train_v_star[epoch + total_epoch_v_star] < tol:
                        print('Successful training v star...')
                        total_epoch_v_star = total_epoch_v_star + epoch + 1
                        v_star_not_done = False

                    elif epoch % 5 == 0 and epoch > 0:
                        if (loss_train_v_star[epoch + total_epoch_v_star] > 
                            loss_train_v_star[epoch + total_epoch_v_star - 1]):
                            mu_v_star = torch.min(torch.tensor((2 * mu_v_star, 1e8)))

                        if (loss_train_v_star[epoch + total_epoch_v_star] <
                            loss_train_v_star[epoch + total_epoch_v_star - 1]):
                            mu_v_star = torch.max(torch.tensor((mu_v_star / 3, 1e-12)))

                    elif (
                        loss_train_v_star[epoch + total_epoch_v_star]
                        / loss_train_v_star[epoch + total_epoch_v_star - 1]
                        > 100
                    ) and epoch > 0:
                        mu_v_star = torch.min(torch.tensor((2 * mu_v_star, 1e8)))

                if u_star_not_done is True and v_star_not_done is True:
                    print(
                        f"epoch = {epoch}, "
                        + "".ljust(13 - len(str(f"epoch = {epoch}, ")))
                        + f"loss_u* = {loss_train_u_star[epoch + total_epoch_u_star]:.2e}, "
                        f"mu_u* = {mu_u_star:.1e}\n"
                        + "".ljust(13)
                        + f"loss_v* = {loss_train_v_star[epoch + total_epoch_v_star]:.2e}, "
                        f"mu_v* = {mu_v_star:.1e}"
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

            # plot loss figure
            fig, axs = plt.subplots(1, 2)
            axs[0].semilogy(
                torch.arange(total_epoch_u_star),
                loss_train_u_star[0:total_epoch_u_star].to(device_cpu),
                label="train",
                linestyle="solid",
            )
            axs[0].semilogy(
                torch.arange(total_epoch_u_star),
                loss_test_u_star[0:total_epoch_u_star].to(device_cpu),
                label="test",
                linestyle="dashed",
            )
            axs[1].semilogy(
                torch.arange(total_epoch_v_star),
                loss_train_v_star[0:total_epoch_v_star].to(device_cpu),
                label="train",
                linestyle="solid",
            )
            axs[1].semilogy(
                torch.arange(total_epoch_v_star),
                loss_test_v_star[0:total_epoch_v_star].to(device_cpu),
                label="test",
                linestyle="dashed",
            )
            axs[0].set_xlabel("Iteration")
            axs[1].set_xlabel("Iteration")
            axs[0].set_ylabel("value of loss function")
            axs[1].set_ylabel("value of loss function")
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")
            axs[0].set_title("Loss of u_star")
            axs[1].set_title("Loss of v_star")
            # plt.savefig(pwd + f"/figure/loss_{int(step*DT)}.png", dpi=300)
            plt.show()

            return params_u_star, params_v_star


class Projection:
    def __init__(self, model, solver, training_p, test_p, *, batch=1):
        self.model = model
        self.solver = solver
        self.batch = batch

        (
            self.training_domain,
            self.training_left_boundary,
            self.training_right_boundary,
            self.training_top_boundary,
            self.training_bottom_boundary,
        ) = training_p
        (
            self.test_domain,
            self.test_left_boundary,
            self.test_right_boundary,
            self.test_top_boundary,
            self.test_bottom_boundary,
        ) = test_p

        self.training_boundary = torch.vstack(
            (
                self.training_left_boundary,
                self.training_right_boundary,
                self.training_top_boundary,
                self.training_bottom_boundary,
            )
        )
        self.test_boundary = torch.vstack(
            (
                self.test_left_boundary,
                self.test_right_boundary,
                self.test_top_boundary,
                self.test_bottom_boundary,
            )
        )

        # Move the data to model device
        self.test_domain = self.test_domain.to(self.model.device)
        self.test_boundary = self.test_boundary.to(self.model.device)

    def main_eq(self, params, points_x, points_y):
        func = (
            self.model.forward_2d_dxx(params, points_x, points_y)
            + 
            self.model.forward_2d_dyy(params, points_x, points_y)
        )
        return func

    def boundary_left_eq(self, params, points_x, points_y):
        func = self.model.forward_2d_dx(params, points_x, points_y)
        return func
    
    def boundary_right_eq(self, params, points_x, points_y):
        func = self.model.forward_2d_dx(params, points_x, points_y)
        return func

    def boundary_top_eq(self, params, points_x, points_y):
        func = self.model.forward_2d_dy(params, points_x, points_y)
        return func
    
    def boundary_bottom_eq(self, params, points_x, points_y):
        func = self.model.forward_2d_dy(params, points_x, points_y)
        return func

    def jacobian_main_eq(self, params, points_x, points_y):
        jacobian = vmap(
            jacrev(self.main_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
        )(params, points_x, points_y)
        row_size = points_x.size(0)
        col_size = torch.sum(
            torch.tensor(
                [torch.add(weight.numel(), bias.numel()) for weight, bias in params]
            )
        )
        jac = torch.zeros(row_size, col_size, device=self.model.device)
        loc = 0
        for idx, (jac_w, jac_b) in enumerate(jacobian):
            jac[:, loc : loc + params[idx][0].numel()] = jac_w.view(
                points_x.size(0), -1
            )
            loc += params[idx][0].numel()
            jac[:, loc : loc + params[idx][1].numel()] = jac_b.view(
                points_x.size(0), -1
            )
            loc += params[idx][1].numel()
        return torch.squeeze(jac)

    def jacobian_boundary_eq(self, params, points_x, points_y):
        left_x, right_x, top_x, bottom_x = torch.tensor_split(points_x, 4)
        left_y, right_y, top_y, bottom_y = torch.tensor_split(points_y, 4)

        jacobian_left = vmap(
            jacrev(self.boundary_left_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
        )(params, left_x, left_y)
        jacobian_right = vmap(
            jacrev(self.boundary_right_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
        )(params, right_x, right_y)
        jacobian_top = vmap(
            jacrev(self.boundary_top_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
        )(params, top_x, top_y)
        jacobian_bottom = vmap(
            jacrev(self.boundary_bottom_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
        )(params, bottom_x, bottom_y)

        row_size = points_x.size(0)
        col_size = torch.sum(
            torch.tensor(
                [torch.add(weight.numel(), bias.numel()) for weight, bias in params]
            )
        )
        jac = torch.zeros(row_size, col_size, device=self.model.device)

        loc = 0
        loc_left = left_x.size(0)
        loc_right = loc_left + right_x.size(0)
        loc_top = loc_right + top_x.size(0)
        loc_bottom = loc_top + bottom_x.size(0)

        for idx, (jac_w_left, jac_b_left, jac_w_right, jac_b_right, 
                  jac_w_top, jac_b_top, jac_w_bottom, jac_b_bottom,
        ) in enumerate(jacobian_left, jacobian_right, jacobian_top, jacobian_bottom):
            jac[0:loc_left, loc : loc + params[idx][0].numel()] = jac_w_left.view(
                left_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[0:loc_left, loc : loc + params[idx][1].numel()] = jac_b_left.view(
                left_x.size(0), -1)
            loc += params[idx][1].numel()
            jac[loc_left:loc_right, loc : loc + params[idx][0].numel()] = jac_w_right.view(
                right_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[loc_left:loc_right, loc : loc + params[idx][1].numel()] = jac_b_right.view(
                right_x.size(0), -1)
            loc += params[idx][1].numel()
            jac[loc_right:loc_top, loc : loc + params[idx][0].numel()] = jac_w_top.view(
                top_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[loc_right:loc_top, loc : loc + params[idx][1].numel()] = jac_b_top.view(
                top_x.size(0), -1)
            loc += params[idx][1].numel()
            jac[loc_top:loc_bottom, loc : loc + params[idx][0].numel()] = jac_w_bottom.view(
                bottom_x.size(0), -1)
            loc += params[idx][0].numel()
            jac[loc_top:loc_bottom, loc : loc + params[idx][1].numel()] = jac_b_bottom.view(
                bottom_x.size(0), -1)
            loc += params[idx][1].numel()
        return torch.squeeze(jac)

    def main_eq_residual(self, params, points_x, points_y, target):
        diff = target - vmap(self.main_eq, in_dims=(None, 0, 0), out_dims=0)(
            params, points_x, points_y
        )
        return diff

    def boundary_eq_residual(self, params, points_x, points_y, target):
        left_x, right_x, top_x, bottom_x = torch.tensor_split(points_x, 4)
        left_y, right_y, top_y, bottom_y = torch.tensor_split(points_y, 4)
        diff = (
            target 
            - 
            torch.vstack((
                vmap(self.boundary_left_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, left_x, left_y),
                vmap(self.boundary_right_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, right_x, right_y),
                vmap(self.boundary_top_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, top_x, top_y),
                vmap(self.boundary_bottom_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, bottom_x, bottom_y)
            ))
        )
        return diff

    def loss_main_eq(self, params, points_x, points_y, target):
        mse = torch.nn.MSELoss(reduction="mean")(
            vmap(self.main_eq, in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y
            ),
            target,
        )
        return mse

    def loss_boundary_eq(self, params, points_x, points_y, target):
        left_x, right_x, top_x, bottom_x = torch.tensor_split(points_x, 4)
        left_y, right_y, top_y, bottom_y = torch.tensor_split(points_y, 4)
        mse = torch.nn.MSELoss(reduction="mean")(
            torch.vstack((
                vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, left_x, left_y),
                vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, right_x, right_y),
                vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, top_x, top_y),
                vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                    params, bottom_x, bottom_y)
            )),
            target,
        )
        return mse


def main():
    neuron_net = [2, 40, 1]

    model_device = device_cpu

    model = Model(net_layers=neuron_net, activation="sigmoid", device=model_device)
    solver = MatrixSolver(device=model_device)

    params_u0 = model.initialize_mlp()
    params_v0 = model.initialize_mlp()
    params_p0 = model.initialize_mlp()
    params_u1 = model.initialize_mlp()
    params_v1 = model.initialize_mlp()

    mesh_gen = CreateMesh()
    training_domain, training_left_bdy, training_right_bdy, \
    training_top_bdy, training_bottom_bdy = mesh_gen.create_mesh(
        50, 50, 50, 50, 500)
    test_domain, test_left_bdy, test_right_bdy, \
    test_top_bdy, test_bottom_bdy = mesh_gen.create_mesh(
        1000, 1000, 1000, 1000, 10000)
    
    # fig, ax = plt.subplots()
    # ax.scatter(training_domain[:, 0].cpu(), training_domain[:, 1].cpu(), c='r')
    # ax.scatter(
    #     torch.vstack(
    #         (
    #             training_left_bdy[:, 0],
    #             training_top_bdy[:, 0],
    #             training_right_bdy[:, 0],
    #             training_bottom_bdy[:, 0],
    #         )
    #     ).cpu(),
    #     torch.vstack(
    #         (
    #             training_left_bdy[:, 1],
    #             training_top_bdy[:, 1],
    #             training_right_bdy[:, 1],
    #             training_bottom_bdy[:, 1],
    #         )
    #     ).cpu(),
    #     c="b",
    # )
    # ax.axis("square")
    # plt.show()
    
    pred = Prediction(
        model,
        solver,
        (
            training_domain,
            training_left_bdy,
            training_right_bdy,
            training_top_bdy,
            training_bottom_bdy,
        ),
        (
            test_domain, 
            test_left_bdy, 
            test_right_bdy, 
            test_top_bdy, 
            test_bottom_bdy),
        batch=1,
    )

    te = 2*DT
    for step, time in enumerate(torch.arange(0, te+DT, DT)):
        print(f'{step}, time = {time}')
        
        if step >= 2:
            # Prediction step
            params_u_star, params_v_star = pred.learning_process(
                params_u0, params_v0, params_p0, params_u1, params_v1, step
            )

            # Projection step

            # convert the data to cpu
            if model.device == device_gpu:
                print("Converting the data to cpu ...")
                params_u_cpu = model.unflatten_params(model.flatten_params(params_u_star).cpu())
                params_v_cpu = model.unflatten_params(model.flatten_params(params_v_star).cpu())
            else:
                params_u_cpu = params_u_star
                params_v_cpu = params_v_star

            # plot the solution
            solution = vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                params_v_cpu, test_domain[:, 0], test_domain[:, 1]
            )
            error = torch.abs(
                exact_sol(test_domain[:, 0], test_domain[:, 1], time, RE, "v") - solution
            )
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            sca = ax.scatter(test_domain[:, 0], test_domain[:, 1], solution, c=solution)
            plt.colorbar(sca)
            plt.show()


if __name__ == "__main__":
    main()