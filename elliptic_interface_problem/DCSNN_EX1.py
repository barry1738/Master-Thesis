import torch
import scipy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, grad, jacrev

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)


class CreateMesh:
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        x = 2.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx) - 1.0
        return x

    
    def boundary_points(self, nx):
        left_x = np.hstack((
            -1.0 * np.ones((nx, 1)),
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0
        ))
        right_x = np.hstack((
            np.ones((nx, 1)),
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0
        ))
        bottom_x = np.hstack((
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            -1.0 * np.ones((nx, 1))
        ))
        top_x = np.hstack((
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            np.ones((nx, 1))
        ))
        x = np.vstack((left_x, right_x, bottom_x, top_x))
        return x
    
    def interface_points(self, nx):
        theta = 2.0 * np.pi * scipy.stats.qmc.LatinHypercube(d=1).random(n=4 * nx)
        x = np.hstack((
            0.2 * np.cos(theta),
            0.5 * np.sin(theta)
        ))
        return x
    
    def sign_x(self, x):
        dist = np.sqrt((x[:, 0] / 0.2) ** 2 + (x[:, 1] / 0.5) ** 2)
        z = np.where(dist < 1.0, -1.0, 1.0)
        return z.reshape(-1, 1)
    
    def normal_vector(self, x):
        """
        Coompute the normal vector of interface points,
        only defined on the interface
        """
        n_x = 2.0 * x[:, 0] / (0.2**2)
        n_y = 2.0 * x[:, 1] / (0.5**2)
        length = np.sqrt(n_x**2 + n_y**2)
        normal_x = n_x / length
        normal_y = n_y / length
        return np.hstack((normal_x, normal_y))
    

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

    def forward(self, x):
        x = self.act(self.ln_in(x))
        for layer in self.hidden:
            x = self.act(layer(x))
        out = self.ln_out(x)
        return out
    

def exact_sol(x, y, z):
    sol1 = torch.sin(x) * torch.sin(y)
    sol2 = torch.exp(x + y)
    sol = sol1 * (1.0 + z) / 2.0 + sol2 * (1.0 - z) / 2.0
    return sol


def compute_loss_Res(func_params, X_inner, Rf_inner):
    out


def main():
    mesh = CreateMesh()
    # interior points
    x_inner = mesh.interior_points(10000)
    # boundary points
    x_bd = mesh.boundary_points(1000)
    # interface points
    x_if = mesh.interface_points(1000)
    print(f"inner_x = {x_inner.shape}")
    print(f"boundary_x = {x_bd.shape}")
    print(f"interface_x = {x_if.shape}")

    sign_z = mesh.sign_x(x_inner)
    print(f'sign_z = {sign_z.shape}')
    normal_vector = mesh.normal_vector(x_if)
    print(f"Normal_vector = {normal_vector.shape}")

    # fig, ax = plt.subplots()
    # ax.scatter(x_inner[:, 0], x_inner[:, 1], c=sign_z, marker='.')
    # ax.scatter(x_bd[:, 0], x_bd[:, 1], c='r', marker='.')
    # ax.scatter(x_if[:, 0], x_if[:, 1], c='g', marker='.')
    # ax.axis('square')
    # plt.show()

    X_inner = np.hstack((x_inner, sign_z))
    X_bd = np.hstack((x_bd, np.ones((x_bd[:, 0].shape[0], 1))))

    X_inner_torch = torch.from_numpy(X_inner).to(device)
    X_bd_torch = torch.from_numpy(X_bd).to(device)
    X_if_torch = torch.from_numpy(x_if).to(device)
    normal_vector_torch = torch.from_numpy(normal_vector).to(device)
    print(f"X_inner_torch = {X_inner_torch.dtype}")

    model = Model(3, [20, 10, 20], 1).to(device)  # hidden layers = [...](list)
    print(model)

    # Make model a functional
    params = dict(model.named_parameters())
    print(functional_call(model, params, X_inner_torch).size())


if __name__ == "__main__":
    main()