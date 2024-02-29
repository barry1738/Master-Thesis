import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vjp, grad

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
    def __init__(self, net_layers, *, activation="sigmoid",
                 rff_spatial=False, device="cpu"):
        # Initialize the network
        self.net_layers = torch.tensor(net_layers)
        self.num_layer = self.net_layers.numel()
        self.rff_spatial = rff_spatial

        # Setting random seed
        self.device = device
        self.g_dev = torch.Generator(device=device)

        # modified net_layers
        if self.rff_spatial is True:
            self.net_layers[0] = (self.net_layers[0]) * 2
        else:
            self.net_layers[0] = self.net_layers[0]

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
        if self.rff_spatial is True:
            input = torch.hstack((
                torch.cos(torch.pi * x), torch.sin(torch.pi * x)
            ))
        else:
            input = x

        for weight, bias in params:
            output = F.linear(input, weight, bias)
            input = self.act(output)
        return output[0]

    def forward_2d(self, params, x, y):
        if self.rff_spatial is True:
            input = torch.hstack((
                torch.cos(x), torch.sin(x), 
                torch.cos(y), torch.sin(y)
            ))
        else:
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
    

class Model_RWF:
    """Random weight factorization model"""

    def __init__(self, net_layers, *, activation="sigmoid",
                 rff_spatial=False, rff_temporal=False, device="cpu"):
        # Initialize the network
        self.net_layers = torch.tensor(net_layers)
        self.num_layer = self.net_layers.numel()
        self.rff_spatial = rff_spatial
        self.rff_temporal = rff_temporal

        # Setting random seed
        self.device = device
        self.g_dev = torch.Generator(device=device)

        # modified net_layers
        if self.rff_spatial is True and self.rff_temporal is False:
            self.net_layers[0] = (self.net_layers[0] - 1) * 2 + 1
        elif self.rff_spatial is True and self.rff_temporal is True:
            self.net_layers[0] = (self.net_layers[0]) * 2
        else:
            self.net_layers[0] = self.net_layers[0]

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
        weights = [multi * torch.randn(n, m, generator=self.g_dev, device=self.device)
                   for (m, n) in zip(self.net_layers[:-1], self.net_layers[1:])]
        biases = [multi * torch.randn(n, generator=self.g_dev, device=self.device) 
                  for n in self.net_layers[1:]]
        return weights, biases

    def random_weight_factorization(self, weights, *, mean=1.0, std=0.1):
        # Random weight factorization
        s = [torch.normal(torch.tensor(mean, device=self.device), 
                          torch.tensor(std, device=self.device) * \
                            torch.eye(size, device=self.device), 
                          generator=self.g_dev) 
             for size in self.net_layers[1:]]
        V = [torch.div(w.t(), torch.diag(torch.exp(ss))).t() 
             for (w, ss) in zip(weights, s)]
        return s, V

    def combine_params(self, s, V, biases):
        # Combine the weights and biases into a single vector
        params = torch.cat(
            [torch.cat((ss.flatten(), vv.flatten(), b.flatten())) 
             for ss, vv, b in zip(s, V, biases)]
        )
        return params.reshape(-1, 1)

    def separate_params(self, params):
        # Separate the combined vector into weights and biases
        s = []
        V = []
        biases = []
        index = 0

        for layer in range(1, self.num_layer):
            # Restore the s matrix
            s_size = self.net_layers[layer] * self.net_layers[layer]
            s_matrix = params[index : index + s_size].reshape(
                self.net_layers[layer], self.net_layers[layer])
            index += s_size

            # Restore the V matrix
            V_size = self.net_layers[layer] * self.net_layers[layer - 1]
            V_matrix = params[index : index + V_size].reshape(
                self.net_layers[layer], self.net_layers[layer - 1]
            )
            index += V_size

            # Restore the bias vector
            bias_size = self.net_layers[layer]
            bias_vector = params[index : index + bias_size].reshape(
                self.net_layers[layer]
            )
            index += bias_size

            s.append(s_matrix)
            V.append(V_matrix)
            biases.append(bias_vector)

        return s, V, biases

    def restore_weights(self, s, V):
        weights = [
            torch.mul(torch.diag(torch.exp(ss)), vv.t()).t() for ss, vv in zip(s, V)
        ]
        return weights

    def forward_1d(self, s, V, biases, x, t):
        if self.rff_spatial is True and self.rff_temporal is False:
            input = torch.hstack((
                torch.cos(torch.pi * x), torch.sin(torch.pi * x), t
            ))
        elif self.rff_spatial is True and self.rff_temporal is True:
            input = torch.hstack((
                torch.cos(torch.pi * x), torch.sin(torch.pi * x), 
                torch.cos(t), torch.sin(t)
            ))
        else:
            input = torch.hstack((x, t))

        # restore the weights
        weights = self.restore_weights(s, V)

        for weight, bias in zip(weights[:-1], biases[:-1]):
            input = self.act(F.linear(input, weight, bias))
        else:
            fc = F.linear(input, weights[-1], biases[-1])
        return fc

    def forward_2d(self, s, V, biases, x, y, t):
        if self.rff_spatial is True and self.rff_temporal is False:
            input = torch.hstack((
                torch.cos(torch.pi * x), torch.sin(torch.pi * x), 
                torch.cos(torch.pi * y), torch.sin(torch.pi * y), t
            ))
        elif self.rff_spatial is True and self.rff_temporal is True:
            input = torch.hstack((
                torch.cos(torch.pi * x), torch.sin(torch.pi * x), 
                torch.cos(torch.pi * y), torch.sin(torch.pi * y), 
                torch.cos(t), torch.sin(t)
            ))
        else:
            input = torch.hstack((x, y, t))

        # restore the weights
        weights = self.restore_weights(s, V)

        for weight, bias in zip(weights[:-1], biases[:-1]):
            input = self.act(F.linear(input, weight, bias))
        else:
            fc = F.linear(input, weights[-1], biases[-1])
        return fc

    def forward_1d_dx(self, s, V, biases, x, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_1d(s, V, biases, primal, t), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_1d_dt(self, s, V, biases, x, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_1d(s, V, biases, x, primal), t
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_1d_dxx(self, s, V, biases, x, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_1d_dx(s, V, biases, primal, t), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_1d_dtt(self, s, V, biases, x, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_1d_dt(s, V, biases, x, primal), t
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dx(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d(s, V, biases, primal, y, t), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dy(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d(s, V, biases, x, primal, t), y
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dt(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d(s, V, biases, x, y, primal), t
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dxx(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d_dx(s, V, biases, primal, y, t), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dxy(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d_dx(s, V, biases, x, primal, t), y
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dyx(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d_dy(s, V, biases, primal, y, t), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dyy(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d_dy(s, V, biases, x, primal, t), y
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_2d_dtt(self, s, V, biases, x, y, t):
        output, vjpfunc = vjp(
            lambda primal: self.forward_2d_dt(s, V, biases, x, y, primal), t
        )
        return vjpfunc(torch.ones_like(output))[0]
