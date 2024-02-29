import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad

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
