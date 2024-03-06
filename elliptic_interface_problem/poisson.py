import torch
import scipy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad, jacrev, vjp

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)


class CreateMesh:
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        x = 2.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx) - 1.0
        return x

    def boundary_points(self, nx):
        left_x = np.hstack(
            (
                -1.0 * np.ones((nx, 1)),
                2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            )
        )
        right_x = np.hstack(
            (
                np.ones((nx, 1)),
                2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            )
        )
        bottom_x = np.hstack(
            (
                2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
                -1.0 * np.ones((nx, 1)),
            )
        )
        top_x = np.hstack(
            (
                2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
                np.ones((nx, 1)),
            )
        )
        x = np.vstack((left_x, right_x, bottom_x, top_x))
        return x


class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.hidden = nn.ModuleList()

        # input layer
        self.ln_in = nn.Linear(in_dim, h_dim[0])

        # hidden layers
        for i in range(len(h_dim) - 1):
            self.hidden.append(nn.Linear(h_dim[i], h_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(h_dim[-1], out_dim)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, x, y):
        input = torch.hstack((x, y))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output


def exact_sol(x):
    sol = torch.sin(2.0 * torch.pi * x[:, 0]) * torch.sin(2.0 * torch.pi * x[:, 1])
    return sol.reshape(-1, 1)


def rhs(x, y):
    f = (
        -4.0
        * torch.pi**2
        * torch.sin(2.0 * torch.pi * x)
        * torch.sin(2.0 * torch.pi * y)
    )
    return f.reshape(-1, 1)


def forward_dx(model, params, x, y):
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dy(model, params, x, y):
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dxx(model, params, x, y):
    output, vjpfunc = vjp(lambda primal: forward_dx(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dyy(model, params, x, y):
    output, vjpfunc = vjp(lambda primal: forward_dy(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]


def compute_loss_Res(model, params, x, y, Rf_inner):
    laplace = forward_dxx(model, params, x, y) + forward_dyy(model, params, x, y)
    loss_Res = laplace - Rf_inner

    return loss_Res.flatten()


def main():
    mesh = CreateMesh()
    # interior points
    x_inner = mesh.interior_points(500)
    # boundary points
    x_bd = mesh.boundary_points(100)
    print(f"inner_x = {x_inner.shape}")
    print(f"boundary_x = {x_bd.shape}")

    X_inner_torch = torch.from_numpy(x_inner).to(device)
    X_bd_torch = torch.from_numpy(x_bd).to(device)

    x_input, y_input = X_inner_torch[:, 0].reshape(-1, 1), X_inner_torch[:, 1].reshape(-1, 1)

    model = Model(2, [40], 1).to(device)  # hidden layers = [...](list)
    print(model)

    # Make model a functional
    u_params = dict(model.named_parameters())
    print(functional_call(model, u_params, (x_input, y_input)).size())

    # Loss function
    loss_Res = compute_loss_Res(model, u_params, x_input, y_input, rhs(x_input, y_input))
    print(f"loss_Res = {loss_Res.size()}")


if __name__ == "__main__":
    main()