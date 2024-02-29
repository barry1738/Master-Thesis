"""
    Poisson equation:

    ∇²u = ∂u/∂x² + ∂u/∂y² = f(x, y)
"""


import os
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.func import jacrev, vmap, grad
from utilities import Model, MatrixSolver
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


def exact_sol(x, y):
    func = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return func


class Poisson:
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

    def boundary_eq(self, params, points_x, points_y):
        func = self.model.forward_2d(params, points_x, points_y)
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
        jacobian = vmap(
            jacrev(self.boundary_eq, argnums=0), in_dims=(None, 0, 0), out_dims=0
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

    def main_eq_residual(self, params, points_x, points_y, target):
        diff = target - vmap(self.main_eq, in_dims=(None, 0, 0), out_dims=0)(
            params, points_x, points_y
        )
        return diff

    def boundary_eq_residual(self, params, points_x, points_y, target):
        diff = target - vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
            params, points_x, points_y
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
        mse = torch.nn.MSELoss(reduction="mean")(
            vmap(self.boundary_eq, in_dims=(None, 0, 0), out_dims=0)(
                params, points_x, points_y
            ),
            target,
        )
        return mse

    def learning_process(self):
        batch_domain = Data.DataLoader(
            dataset=Data.TensorDataset(
                self.training_domain[:, 0], self.training_domain[:, 1]
            ),
            batch_size=int(self.training_domain[:, 0].size(0) / self.batch),
            shuffle=True,
            drop_last=True,
        )
        batch_boundary = Data.DataLoader(
            dataset=Data.TensorDataset(
                self.training_boundary[:, 0], self.training_boundary[:, 1]
            ),
            batch_size=int(self.training_boundary[:, 0].size(0) / self.batch),
            shuffle=True,
            drop_last=True,
        )

        params_u = self.model.initialize_mlp()

        # Initialize the parameters
        Niter = 10**4
        tol = 1e-10
        total_epoch = 0
        # Initialize the loss function
        loss_train = torch.zeros(self.batch * Niter, device=self.model.device)
        loss_test = torch.zeros(self.batch * Niter, device=self.model.device)

        for batch_step, (data_domain, data_boundary) in enumerate(
            zip(batch_domain, batch_boundary)
        ):
            print(f"Batch step = {batch_step}")

            # Move the data to GPU
            for idx in torch.arange(2):
                data_domain[idx] = data_domain[idx].to(self.model.device)
                data_boundary[idx] = data_boundary[idx].to(self.model.device)

            # Initialize the parameters
            mu = torch.tensor(10**3, device=self.model.device)
            alpha = torch.tensor(1.0, device=self.model.device)
            beta = torch.tensor(1.0, device=self.model.device)

            rhs_u = (
                vmap(grad(grad(exact_sol, argnums=0), argnums=0), 
                     in_dims=(0, 0), out_dims=0)(
                         data_domain[0], data_domain[1]
                     )
                +
                vmap(grad(grad(exact_sol, argnums=1), argnums=1),
                     in_dims=(0, 0), out_dims=0)(
                         data_domain[0], data_domain[1]
                     )
            )
            rhs_u_test = (
                vmap(grad(grad(exact_sol, argnums=0), argnums=0),
                     in_dims=(0, 0), out_dims=0)(
                         self.test_domain[:, 0], self.test_domain[:, 1]
                    )
                +
                vmap(grad(grad(exact_sol, argnums=1), argnums=1),
                     in_dims=(0, 0), out_dims=0)(
                        self.test_domain[:, 0], self.test_domain[:, 1]
                    )
            )

            for epoch in range(Niter):
                # Compute each jacobian matrix
                jac_main_eq = self.jacobian_main_eq(
                    params_u, data_domain[0], data_domain[1]
                )
                jac_boundary_eq = self.jacobian_boundary_eq(
                    params_u, data_boundary[0], data_boundary[1]
                )
                # Combine the jacobian matrix
                jacobian = torch.vstack(
                    (
                        torch.sqrt(alpha / torch.tensor(data_domain[0].size(0), 
                                                        device=self.model.device))
                        * jac_main_eq,
                        torch.sqrt(beta / torch.tensor(data_boundary[0].size(0), 
                                                       device=self.model.device))
                        * jac_boundary_eq
                    )
                )
                # print(f'jacobian size = {jacobian.size()}')

                # Compute each residual
                residual_main_eq = self.main_eq_residual(
                    params_u, 
                    data_domain[0], 
                    data_domain[1], 
                    rhs_u
                )
                residual_boundary_eq = self.boundary_eq_residual(
                    params_u,
                    data_boundary[0],
                    data_boundary[1],
                    torch.zeros(data_boundary[0].size(0), device=self.model.device)
                )
                # Combine the residual
                residual = torch.hstack(
                    (
                        residual_main_eq
                        * torch.sqrt(alpha / torch.tensor(data_domain[0].size(0))),
                        residual_boundary_eq
                        * torch.sqrt(beta / torch.tensor(data_boundary[0].size(0))),
                    )
                )
                # print(f'residual size = {residual.size()}')

                # Compute the update value of weight and bias
                # h = self.solver.cholesky(jacobian, diff, mu)
                h = self.solver.qr_decomposition(jacobian, residual.view(-1, 1), mu)
                # print(f'h size = {h.size()}')
                params_u_flatten = self.model.flatten_params(params_u)
                params_u_flatten = torch.add(params_u_flatten, h)
                params_u = self.model.unflatten_params(params_u_flatten)

                # Compute the loss function
                loss_train[epoch + total_epoch] = (
                    alpha * self.loss_main_eq(
                        params_u, 
                        data_domain[0], 
                        data_domain[1], 
                        rhs_u
                    )
                    + 
                    beta * self.loss_boundary_eq(
                        params_u,
                        data_boundary[0],
                        data_boundary[1],
                        torch.zeros(data_boundary[0].size(0), 
                                    device=self.model.device)
                    )
                )

                loss_test[epoch + total_epoch] = (
                    self.loss_main_eq(
                        params_u,
                        self.test_domain[:, 0],
                        self.test_domain[:, 1],
                        rhs_u_test
                    )
                    + self.loss_boundary_eq(
                        params_u,
                        self.test_boundary[:, 0],
                        self.test_boundary[:, 1],
                        torch.zeros(self.test_boundary[:, 0].size(0), 
                                    device=self.model.device)
                    )
                )

                # Update alpha and beta
                if epoch % 1000 == 0:
                    residual_params_alpha = grad(self.loss_main_eq, argnums=0)(
                        params_u, 
                        data_domain[0], 
                        data_domain[1], 
                        rhs_u
                    )
                    loss_domain_deriv_params = torch.linalg.vector_norm(
                        self.model.flatten_params(residual_params_alpha)
                    )
                    residual_params_beta = grad(self.loss_boundary_eq, argnums=0)(
                        params_u,
                        data_boundary[0],
                        data_boundary[1],
                        torch.zeros(data_boundary[0].size(0), device=self.model.device)
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

                    alpha = (1 - 0.1) * alpha + 0.1 * alpha_bar
                    beta = (1 - 0.1) * beta + 0.1 * beta_bar
                    print(f"alpha = {alpha:.3f}, beta = {beta:.3f}")

                print(
                    f"epoch = {epoch}, "
                    f"loss_train = {loss_train[epoch + total_epoch]:.2e}, "
                    f"mu = {mu:.1e}"
                )

                # Stop this batch if the loss function is small enough
                if loss_train[epoch + total_epoch] < tol:
                    total_epoch = total_epoch + epoch + 1
                    print("Successful training ...")
                    break
                elif epoch % 5 == 0 and epoch > 0:
                    if (
                        loss_train[epoch + total_epoch]
                        > loss_train[epoch + total_epoch - 1]
                    ):
                        mu = torch.min(torch.tensor((2 * mu, 1e8)))

                    if (
                        loss_train[epoch + total_epoch]
                        < loss_train[epoch + total_epoch - 1]
                    ):
                        mu = torch.max(torch.tensor((mu / 3, 1e-12)))
                elif (
                    loss_train[epoch + total_epoch]
                    / loss_train[epoch + total_epoch - 1]
                    > 100
                ) and epoch > 0:
                    mu = torch.min(torch.tensor((2 * mu, 1e8)))

                # if (
                #     loss_train[epoch + total_epoch] > 1e13
                #     or loss_train[epoch + total_epoch]
                #     == loss_train[epoch + total_epoch - 1]
                # ):
                #     print("Failed to learn ...")
                #     return params_u_star

            # plot loss figure
            fig, ax = plt.subplots()
            ax.semilogy(
                torch.arange(total_epoch),
                loss_train[0:total_epoch].to(device_cpu),
                label="train",
                linestyle="solid",
            )
            ax.semilogy(
                torch.arange(total_epoch),
                loss_test[0:total_epoch].to(device_cpu),
                label="test",
                linestyle="dashed",
            )
            plt.xlabel("Iteration")
            plt.ylabel("value of loss function")
            plt.legend(loc="upper right")
            plt.title("Loss")
            # plt.savefig(pwd + f"/figure/loss_{int(step*DT)}.png", dpi=300)
            plt.show()

            return params_u


def main():
    neuron_net = [2, 40, 1]
    model_device = device_cpu

    model = Model(
        net_layers=neuron_net, 
        activation="sigmoid", 
        device=model_device
    )
    solver = MatrixSolver(device=model_device)

    mesh_gen = CreateMesh()
    (
        training_domain,
        training_left_bdy,
        training_right_bdy,
        training_top_bdy,
        training_bottom_bdy,
    ) = mesh_gen.create_mesh(50, 50, 50, 50, 1000)
    (
        test_domain,
        test_left_bdy,
        test_right_bdy,
        test_top_bdy,
        test_bottom_bdy,
    ) = mesh_gen.create_mesh(1000, 1000, 1000, 1000, 10000)

    # fig, ax = plt.subplots()
    # ax.scatter(training_domain[:, 0].cpu(), training_domain[:, 1].cpu(), c="r")
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

    pred = Poisson(
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
            test_bottom_bdy
        ),
        batch=1,
    )

    params_u = pred.learning_process()

    # convert the data to cpu
    if model.device == device_gpu:
        print("Converting the data to cpu ...")
        params_u_cpu = model.unflatten_params(model.flatten_params(params_u).cpu())
    else:
        params_u_cpu = params_u

    # plot the solution
    solution = vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
        params_u_cpu, test_domain[:, 0], test_domain[:, 1]
    )
    error = torch.abs(exact_sol(test_domain[:, 0], test_domain[:, 1]) - solution)
    fig, ax = plt.subplots()
    sca = ax.scatter(test_domain[:, 0], test_domain[:, 1], c=error)
    plt.colorbar(sca)
    ax.axis("square")
    plt.show()


if __name__ == "__main__":
    main()