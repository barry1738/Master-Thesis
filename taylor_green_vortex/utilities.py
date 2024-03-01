import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

torch.set_default_dtype(torch.double)

class MatrixSolver:
    def __init__(self, device="cpu"):
        self.device = device

    def cholesky(self, J, diff, mu):
        # Solve the linear system using Cholesky decomposition
        A = J.t() @ J + mu * torch.eye(J.shape[1], device=self.device)
        b = J.t() @ diff
        L = torch.linalg.cholesky(A)
        y = torch.linalg.solve_triangular(L, b, upper=False)
        x = torch.linalg.solve_triangular(L.t(), y, upper=True)
        return x

    def svd(self, J, diff, mu):
        # Solve the linear system using SVD
        u, s, vh = torch.linalg.svd(J, full_matrices=False)
        return vh.t() @ torch.diag(s / (s**2 + mu)) @ u.t() @ diff

    def qr_decomposition(self, J, diff, mu):
        # Solve the linear system using QR decomposition
        A = torch.vstack((J, mu**0.5 * torch.eye(J.size(1), device=self.device)))
        b = torch.vstack((diff, torch.zeros(J.size(1), 1, device=self.device)))
        Q, R = torch.linalg.qr(A)
        x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
        return x


class Model:
    def __init__(self, net_layers, *, activation="sigmoid", device="cpu"):
        # Initialize the network
        self.net_layers = torch.tensor(net_layers)
        self.num_layer = self.net_layers.numel()

        # Setting random seed
        self.device = device
        self.g_dev = torch.Generator(device=device)

        # Decide which activation function to use
        match activation:
            case "sigmoid":
                self.act = nn.Sigmoid()
                print("Using sigmoid as activation function.")
            case "swish":
                self.act = nn.SiLU()
                print("Using swish as activation function.")
            case _:
                self.act = nn.Sigmoid()
                print("Using sigmoid as activation function.")

    def initialize_mlp(self, *, multi=1.0):
        # Initialize the weights and biases
        params = [(multi * torch.randn(n, m, generator=self.g_dev, device=self.device),
                   multi * torch.randn(n, generator=self.g_dev, device=self.device))
                   for (m, n) in zip(self.net_layers[:-1], self.net_layers[1:])]
        return params

    def flatten_params(self, params):
        # Combine the weights and biases into a single vector
        params_flatten = torch.cat(
            [torch.cat((w.flatten(), b.flatten())) for (w, b) in params]
        )
        return params_flatten.reshape(-1, 1)
    
    def unflatten_params(self, params_flatten):
        # Separate the combined vector into weights and biases
        params = []
        index = 0
        for layer in range(1, self.num_layer):
            # Restore the weight matrix
            weight_size = self.net_layers[layer] * self.net_layers[layer - 1]
            weight_matrix = params_flatten[index : index + weight_size].reshape(
                self.net_layers[layer], self.net_layers[layer - 1]
            )
            index += weight_size

            # Restore the bias vector
            bias_size = self.net_layers[layer]
            bias_vector = params_flatten[index : index + bias_size].reshape(
                self.net_layers[layer]
            )
            index += bias_size

            params.append((weight_matrix, bias_vector))

        return params

    def forward_1d(self, params, x):
        input = x
        for weight, bias in params:
            output = F.linear(input, weight, bias)
            input = self.act(output)
        return output[0]

    def forward_2d(self, params, x, y):
        input = torch.hstack((x, y))

        for weight, bias in params:
            output = F.linear(input, weight, bias)
            input = self.act(output)
        return output[0]

    def forward_1d_dx(self, params, x):
        diff_x = grad(self.forward_1d, 1)(params, x)
        return diff_x
    
    def forward_1d_dxx(self, params, x):
        diff2_x = grad(grad(self.forward_1d, 1), 1)(params, x)
        return diff2_x

    def forward_2d_dx(self, params, x, y):
        diff_x = grad(self.forward_2d, 1)(params, x, y)
        return diff_x

    def forward_2d_dy(self, params, x, y):
        diff_y = grad(self.forward_2d, 2)(params, x, y)
        return diff_y

    def forward_2d_dxx(self, params, x, y):
        diff2_x = grad(grad(self.forward_2d, 1), 1)(params, x, y)
        return diff2_x

    def forward_2d_dyy(self, params, x, y):
        diff2_y = grad(grad(self.forward_2d, 2), 2)(params, x, y)
        return diff2_y

    def mse_loss(self, diff):
        # Calculate the mean square error
        mse = torch.sum(torch.square(diff))
        return mse


def exact_sol(x, y, t, Re, type):
    """calculate analytical solution
    u(x,y,t) = -cos(πx)sin(πy)exp(-2π²t/RE)
    v(x,y,t) =  sin(πx)cos(πy)exp(-2π²t/RE)
    p(x,y,t) = -0.25(cos(2πx)+cos(2πy))exp(-4π²t/RE)
    """
    match type:
        case "u":
            exp_val = torch.tensor([-2 * torch.pi**2 * t / Re])
            func = (
                -torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(exp_val)
            )
            return func[0]
        case "v":
            exp_val = torch.tensor([-2 * torch.pi**2 * t / Re])
            func = (
                torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(exp_val)
            )
            return func[0]
        case "p":
            exp_val = torch.tensor([-4 * torch.pi**2 * t / Re])
            func = (
                -0.25
                * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))
                * torch.exp(exp_val)
            )
            return func[0]
        

def compute_u_star_rhs(model, params_u0, params_u1, params_v0, params_v1, 
                       params_p1, points_x, points_y, step, *, Dt=0.01, Re=400.0):
    """ Compute the right-hand side of the u_star equation """

    # Compute the right-hand side of the u_star equation
    if step == 2:
        u_star_rhs = (
            + 4 * vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, Dt, Re, "u")
            - vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, 0.0, Re, "u")
            - 2 * (2 * Dt) * (
                  vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                * vmap(grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None),out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
            )
            + (2 * Dt) * (
                  vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "u")
                * vmap( grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "u")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "u")
            )
            - (2 * Dt) * (
                vmap(grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "p")
            )
        )

        v_star_rhs = (
            + 4 * vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, Dt, Re, "v")
            - vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, 0.0, Re, "v")
            - 2 * (2 * Dt) * (
                  vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                * vmap(grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
            )
            + (2 * Dt) * (
                  vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "u")
                * vmap(grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "v")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, 0.0, Re, "v")
            )
            - (2 * Dt) * (
                vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "p")
            )
        )

    elif step == 3:
        u_star_rhs = (
            + 4 * vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                params_u1, points_x, points_y)
            - vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, Dt, Re, "u")
            - 2 * (2 * Dt) * (
                  vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_u1, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dims=(None, 0, 0), out_dims=0)(
                    params_u1, points_x, points_y)
                + vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_v1, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dims=(None, 0, 0), out_dims=0)(
                    params_u1, points_x, points_y)
            )
            + (2 * Dt) * (
                vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                * vmap( grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
            )
            - (2 * Dt) * (
                vmap(model.forward_2d_dx, in_dims=(None, 0, 0), out_dims=0)(
                    params_p1, points_x, points_y)
            )
        )

        v_star_rhs = (
            + 4 * vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                params_v1, points_x, points_y)
            - vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                points_x, points_y, Dt, Re, "v")
            - 2 * (2 * Dt) * (
                  vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_u1, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dim=(None, 0, 0), out_dim=0)(
                    params_v1, points_x, points_y)
                + vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_v1, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dim=(None, 0, 0), out_dim=0)(
                    params_v1, points_x, points_y)
            )
            + (2 * Dt) * (
                  vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "u")
                * vmap(grad(exact_sol, 0), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                + vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
                * vmap(grad(exact_sol, 1), in_dims=(0, 0, None, None, None), out_dims=0)(
                    points_x, points_y, Dt, Re, "v")
            )
            - (2 * Dt) * (
                vmap(model.forward_2d_dy, in_dim=(None, 0, 0), out_dim=0)(
                    params_p1, points_x, points_y)
            )
        )

    else:
        u_star_rhs = (
            + 4 * vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                params_u1, points_x, points_y)
            - vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                params_u0, points_x, points_y)
            - 2 * (2 * Dt) * (
                vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_u1, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dim=(None, 0, 0), out_dim=0)(
                    params_u1, points_x, points_y)
                + vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_v1, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dim=(None, 0, 0), out_dim=0)(
                    params_u1, points_x, points_y)
            )
            + (2 * Dt) * (
                vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_u0, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dim=(None, 0, 0), out_dim=0)(
                    params_u0, points_x, points_y)
                + vmap(model.forward_2d, in_dim=(None, 0, 0), out_dim=0)(
                    params_v0, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dim=(None, 0, 0), out_dim=0)(
                    params_u0, points_x, points_y)
            )
            - (2 * Dt) * (
                vmap(model.forward_2d_dx, in_dim=(None, 0, 0), out_dim=0)(
                    params_p1, points_x, points_y)
            )
        )

        v_star_rhs = (
            + 4 * vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                params_v1, points_x, points_y)
            - vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                params_v0, points_x, points_y)
            - 2 * (2 * Dt) * (
                vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_u1, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dims=(None, 0, 0), out_dims=0)(
                    params_v1, points_x, points_y)
                + vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_v1, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dims=(None, 0, 0), out_dims=0)(
                    params_v1, points_x, points_y)
            )
            + (2 * Dt) * (
                vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_u0, points_x, points_y)
                * vmap(model.forward_2d_dx, in_dims=(None, 0, 0), out_dims=0)(
                    params_v0, points_x, points_y)
                + vmap(model.forward_2d, in_dims=(None, 0, 0), out_dims=0)(
                    params_v0, points_x, points_y)
                * vmap(model.forward_2d_dy, in_dims=(None, 0, 0), out_dims=0)(
                    params_v0, points_x, points_y)
            )
            - (2 * Dt) * (
                vmap(model.forward_2d_dy, in_dims=(None, 0, 0), out_dims=0)(
                    params_p1, points_x, points_y)
            )
        )
    
    return u_star_rhs, v_star_rhs


def compute_u_star_bdy_value(points_x, points_y, time, Re):
    """ Compute the boundary value of u_star """

    # Compute the boundary value of u_star
    u_star_bdy_rhs = vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
        points_x, points_y, time, Re, "u")
    v_star_bdy_rhs = vmap(exact_sol, in_dims=(0, 0, None, None, None), out_dims=0)(
        points_x, points_y, time, Re, "v")

    return u_star_bdy_rhs, v_star_bdy_rhs