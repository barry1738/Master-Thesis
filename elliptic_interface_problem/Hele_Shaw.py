import os
import time
import torch
import numpy as np
import torch.nn as nn
import scipy.stats.qmc as qmc
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, vjp, jacrev, grad


torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)


# pwd = "/home/barry/Desktop/2024_03_18/"
pwd = "C:\\Users\\barry\\Desktop\\2024_03_18\\"
# dir_name = "cos_6t/"
dir_name = "cos_5t\\"
if not os.path.exists(pwd + dir_name):
    print("Creating data directory...")
    os.makedirs(pwd + dir_name)
else:
    print("Data directory already exists...")


class CreateMesh:
    def __init__(self, interface_func, *, radius=1):
        self.func = interface_func
        self.r = radius

    # def domain_points(self, n, *, xc=0, yc=0):
    #     """Uniform random distribution within a circle"""
    #     radius = torch.tensor(self.r * np.sqrt(qmc.LatinHypercube(d=1).random(n=n)))
    #     theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
    #     x = xc + radius * torch.cos(theta)
    #     y = yc + radius * torch.sin(theta)
    #     return x, y

    def domain_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution within a circle"""
        nx = n // 5
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=1*nx))
        theta_if = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=4*nx))
        radius = torch.tensor(self.r * np.sqrt(qmc.LatinHypercube(d=1).random(n=1*nx)))
        radius_if = self.func(theta_if)
        x_eps = torch.Tensor(4*nx, 1).uniform_(-0.2, 0.2)
        y_eps = torch.Tensor(4*nx, 1).uniform_(-0.2, 0.2)
        x = xc + radius * torch.cos(theta)
        y = yc + radius * torch.sin(theta)
        x_if = xc + (radius_if + x_eps) * torch.cos(theta_if)
        y_if = yc + (radius_if + y_eps) * torch.sin(theta_if)
        return torch.vstack((x, x_if)), torch.vstack((y, y_if))
        # return x_if, y_if
    
    def boundary_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a circle"""
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
        x = xc + self.r * torch.cos(theta)
        y = yc + self.r * torch.sin(theta)
        return x, y

    def interface_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a polar curve"""
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
        radius = self.func(theta)
        x = xc + radius * torch.cos(theta)
        y = yc + radius * torch.sin(theta)
        return x, y

    def sign(self, x, y):
        """Check the points inside the polar curve, return z = -1 if inside, z = 1 if outside"""
        dist = torch.sqrt(x ** 2 + y ** 2) - self.func(torch.atan2(y, x))
        z = torch.where(dist > 0, 1, -1)
        return z
    
    def compute_boundary_normal_vec(self, x, y):
        """Compute the boundary normal vector"""
        nx = 2 * x / 1
        ny = 2 * y / 1
        dist = torch.sqrt(nx ** 2 + ny ** 2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny
    
    def compute_interface_normal_vec(self, x, y):
        """Compute the interface normal vector"""
        theta = torch.atan2(y, x)
        r = self.func(theta)
        drdt = vmap(grad(self.func))(theta.reshape(-1)).view(-1, 1)
        nx = drdt * torch.sin(theta) + r * torch.cos(theta)
        ny = -drdt * torch.cos(theta) + r * torch.sin(theta)
        dist = torch.sqrt(nx ** 2 + ny ** 2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny
    
    def compute_interface_curvature(self, x, y):
        """Compute the interface curvature"""
        theta = torch.atan2(y, x)
        r = self.func(theta)
        drdt = vmap(grad(self.func))(theta.reshape(-1)).view(-1, 1)
        d2rdt2 = vmap(grad(grad(self.func)))(theta.reshape(-1)).view(-1, 1)
        dxdt = drdt * torch.cos(theta) - r * torch.sin(theta)
        dydt = drdt * torch.sin(theta) + r * torch.cos(theta)
        d2xdt2 = d2rdt2 * torch.cos(theta) - 2 * drdt * torch.sin(theta) - r * torch.cos(theta)
        d2ydt2 = d2rdt2 * torch.sin(theta) + 2 * drdt * torch.cos(theta) - r * torch.sin(theta)
        curvature = (dxdt * d2ydt2 - dydt * d2xdt2) / (dxdt ** 2 + dydt ** 2) ** (3 / 2)
        return curvature
    

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        # input layer
        self.ln_in = nn.Linear(in_dim, h_dim[0])

        self.hidden = nn.ModuleList()
        # hidden layers
        for i in range(len(h_dim) - 1):
            self.hidden.append(nn.Linear(h_dim[i], h_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(h_dim[-1], out_dim, bias=True)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()
        # self.act = nn.SiLU()

    def forward(self, x, y, z):
        input = torch.hstack((x, y, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

def forward_dx(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (primal, y, z)), x
    )
    return vjpfunc(torch.ones_like(output))[0]


def forward_dy(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (x, primal, z)), y
    )
    return vjpfunc(torch.ones_like(output))[0]


def forward_dxx(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: forward_dx(model, params, primal, y, z), x)
    return vjpfunc(torch.ones_like(output))[0]


def forward_dyy(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: forward_dy(model, params, x, primal, z), y)
    return vjpfunc(torch.ones_like(output))[0]


def compute_loss_res(model, params, x, y, z, Rf_inner):
    laplace_pred = forward_dxx(model, params, x, y, z) + forward_dyy(model, params, x, y, z)
    loss = laplace_pred - Rf_inner
    return loss


def compute_loss_bd(model, params, x, y, z, nx, ny, Rf_bd):
    dfdx = forward_dx(model, params, x, y, z)
    dfdy = forward_dy(model, params, x, y, z)
    normal_pred = dfdx * nx + dfdy * ny
    loss = normal_pred - Rf_bd
    # pred = functional_call(model, params, (x, y, z))
    # loss = pred - Rf_bd
    return loss


def compute_loss_if_1(model, params, x, y, Rf_if_1):
    z_inner = -1.0 * torch.ones_like(x)
    z_outer = 1.0 * torch.ones_like(x)
    pred_inner = functional_call(model, params, (x, y, z_inner))
    pred_outer = functional_call(model, params, (x, y, z_outer))
    pred = pred_outer - pred_inner
    loss = pred - Rf_if_1
    return loss


def compute_loss_if_2(model, params, x, y, nx, ny, Rf_if_2, beta):
    z_inner = -1.0 * torch.ones_like(x)
    z_outer = 1.0 * torch.ones_like(x)
    dfdx_outer = forward_dx(model, params, x, y, z_outer)
    dfdy_outer = forward_dy(model, params, x, y, z_outer)
    dfdx_inner = forward_dx(model, params, x, y, z_inner)
    dfdy_inner = forward_dy(model, params, x, y, z_inner)
    pred_outer = nx * dfdx_outer + ny * dfdy_outer
    pred_inner = nx * dfdx_inner + ny * dfdy_inner
    pred = pred_outer / beta - pred_inner
    loss = pred - Rf_if_2
    return loss


def qr_decomposition(J_mat, diff, mu):
    """Solve the linear system using QR decomposition"""
    A = torch.vstack((J_mat, mu**0.5 * torch.eye(J_mat.size(1), device=device)))
    b = torch.vstack((-diff, torch.zeros(J_mat.size(1), 1, device=device)))
    Q, R = torch.linalg.qr(A)
    x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
    return x.flatten()


def cholesky(J, diff, mu):
    """Solve the linear system using Cholesky decomposition"""
    A = J.t() @ J + mu * torch.eye(J.shape[1], device=device)
    b = J.t() @ -diff
    L = torch.linalg.cholesky(A)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    x = torch.linalg.solve_triangular(L.t(), y, upper=True)
    return x.flatten()


def main():
    # Define the model
    model = Model(3, [20, 20, 20, 20], 1).to(device)
    # print(model)

    # Create the training data
    mesh = CreateMesh(interface_func=lambda t: 1 + 0.1 * torch.cos(5 * t), radius=1.5)
    x_inner, y_inner = mesh.domain_points(3000)
    x_bd, y_bd = mesh.boundary_points(100)
    x_if, y_if = mesh.interface_points(200)
    z_inner = mesh.sign(x_inner, y_inner)
    z_bd = torch.ones_like(x_bd)

    # Create the validation data
    x_inner_v, y_inner_v = mesh.domain_points(10000)
    x_bd_v, y_bd_v = mesh.boundary_points(1000)
    x_if_v, y_if_v = mesh.interface_points(1000)
    z_inner_v = mesh.sign(x_inner_v, y_inner_v)
    z_bd_v = torch.ones_like(x_bd_v)

    print(f"Number of x_inner = {x_inner.shape}")
    print(f"Number of x_bd = {x_bd.shape}")
    print(f"Number of x_if = {x_if.shape}")

    # Compute the boundary normal vector
    nx_bd, ny_bd = mesh.compute_boundary_normal_vec(x_bd, y_bd)
    nx_if, ny_if = mesh.compute_interface_normal_vec(x_if, y_if)
    nx_bd_v, ny_bd_v = mesh.compute_boundary_normal_vec(x_bd_v, y_bd_v)
    nx_if_v, ny_if_v = mesh.compute_interface_normal_vec(x_if_v, y_if_v)

    # Compute the interface curvature
    k_if = mesh.compute_interface_curvature(x_if, y_if)
    k_if_v = mesh.compute_interface_curvature(x_if_v, y_if_v)
    # Compute the length of the interface points from the origin
    r_if = torch.sqrt(x_if ** 2 + y_if ** 2)
    r_if_v = torch.sqrt(x_if_v ** 2 + y_if_v ** 2)

    # Plot the training data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(x_inner, y_inner, c=z_inner, s=1)
    ax[0].scatter(x_bd, y_bd, s=1)
    sca = ax[0].scatter(x_if, y_if, c=k_if, s=1)
    # ax.quiver(x_bd, y_bd, nx_bd, ny_bd, angles="xy", scale_units="xy", scale=5)
    # ax.quiver(x_if, y_if, nx_if, ny_if, angles="xy", scale_units="xy", scale=5)
    ax[0].axis('square')
    ax[1].scatter(torch.arctan2(y_if, x_if), k_if, s=5)
    ax[0].set_title("Training Data")
    ax[1].set_title("Interface Curvature")
    plt.colorbar(sca)
    plt.savefig(pwd + dir_name + "training_data.png", dpi=300)
    plt.show()

    # Move the data to the device
    x_inner, y_inner, z_inner = x_inner.to(device), y_inner.to(device), z_inner.to(device)
    x_bd, y_bd, z_bd = x_bd.to(device), y_bd.to(device), z_bd.to(device)
    x_if, y_if = x_if.to(device), y_if.to(device)
    nx_bd, ny_bd = nx_bd.to(device), ny_bd.to(device)
    nx_if, ny_if = nx_if.to(device), ny_if.to(device)
    k_if = k_if.to(device)
    r_if = r_if.to(device)

    x_inner_v, y_inner_v, z_inner_v = x_inner_v.to(device), y_inner_v.to(device), z_inner_v.to(device)
    x_bd_v, y_bd_v, z_bd_v = x_bd_v.to(device), y_bd_v.to(device), z_bd_v.to(device)
    x_if_v, y_if_v = x_if_v.to(device), y_if_v.to(device)
    nx_bd_v, ny_bd_v = nx_bd_v.to(device), ny_bd_v.to(device)
    nx_if_v, ny_if_v = nx_if_v.to(device), ny_if_v.to(device)
    k_if_v = k_if_v.to(device)
    r_if_v = r_if_v.to(device)

    # get the training parameters and total number of parameters
    u_params = dict(model.named_parameters())
    # 10 times the initial parameters
    u_params_flatten = 10.0 * nn.utils.parameters_to_vector(u_params.values())
    nn.utils.vector_to_parameters(u_params_flatten, u_params.values())
    print(f"Number of parameters = {u_params_flatten.numel()}")

    # Define the right-hand side vector
    Ca = torch.tensor(100)
    beta = torch.tensor(100)
    Rf_inner = torch.zeros_like(x_inner)
    Rf_bd = torch.zeros_like(x_bd)
    Rf_if_1 = -k_if / Ca + (1 - 1 / beta) * torch.log(r_if) / (2 * torch.pi)
    Rf_if_2 = torch.zeros_like(x_if)

    Rf_inner_v = torch.zeros_like(x_inner_v)
    Rf_bd_v = torch.zeros_like(x_bd_v)
    Rf_if_1_v = -k_if_v / Ca + (1 - 1 / beta) * torch.log(r_if_v) / (2 * torch.pi)
    Rf_if_2_v = torch.zeros_like(x_if_v)

    # Start the training
    Niter = 1000
    tol = 1.0e-9
    mu = 1.0e3
    alpha_res = 1.0
    alpha_bd = 1.0
    alpha_if_1 = 1.0
    alpha_if_2 = 1.0
    savedloss = []
    saveloss_vaild = []

    start_time = time.time()
    for step in range(Niter):
        # Compute the Jacobian matrix
        jac_res_dict = vmap(
            jacrev(compute_loss_res, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0),
            out_dims=0,
        )(model, u_params, x_inner, y_inner, z_inner, Rf_inner)

        jac_bd_dict = vmap(
            jacrev(compute_loss_bd, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0, 0, 0),
            out_dims=0,
        )(model, u_params, x_bd, y_bd, z_bd, nx_bd, ny_bd, Rf_bd)

        jac_if_1_dict = vmap(
            jacrev(compute_loss_if_1, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, x_if, y_if, Rf_if_1)

        jac_if_2_dict = vmap(
            jacrev(compute_loss_if_2, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0, 0, None),
            out_dims=0,
        )(model, u_params, x_if, y_if, nx_if, ny_if, Rf_if_2, beta)

        jac_res = torch.hstack([v.view(x_inner.size(0), -1) for v in jac_res_dict.values()])
        jac_bd = torch.hstack([v.view(x_bd.size(0), -1) for v in jac_bd_dict.values()])
        jac_if_1 = torch.hstack([v.view(x_if.size(0), -1) for v in jac_if_1_dict.values()])
        jac_if_2 = torch.hstack([v.view(x_if.size(0), -1) for v in jac_if_2_dict.values()])

        Jac_res = jac_res * torch.sqrt(alpha_res / torch.tensor(x_inner.size(0)))
        Jac_bd = jac_bd * torch.sqrt(alpha_bd / torch.tensor(x_bd.size(0)))
        Jac_if_1 = jac_if_1 * torch.sqrt(alpha_if_1 / torch.tensor(x_if.size(0)))
        Jac_if_2 = jac_if_2 * torch.sqrt(alpha_if_2 / torch.tensor(x_if.size(0)))
        # print(f"Jac_res.shape = {Jac_res.shape}")
        # print(f"Jac_bd.shape = {Jac_bd.shape}")
        # print(f"Jac_if_1.shape = {Jac_if_1.shape}")
        # print(f"Jac_if_2.shape = {Jac_if_2.shape}")

        # Compute the residual vector
        l_vec_res = compute_loss_res(model, u_params, x_inner, y_inner, z_inner, Rf_inner)
        l_vec_bd = compute_loss_bd(model, u_params, x_bd, y_bd, z_bd, nx_bd, ny_bd, Rf_bd)
        l_vec_if_1 = compute_loss_if_1(model, u_params, x_if, y_if, Rf_if_1)
        l_vec_if_2 = compute_loss_if_2(model, u_params, x_if, y_if, nx_if, ny_if, Rf_if_2, beta)
        l_vec_res_v = compute_loss_res(model, u_params, x_inner_v, y_inner_v, z_inner_v, Rf_inner_v)
        l_vec_bd_v = compute_loss_bd(model, u_params, x_bd_v, y_bd_v, z_bd_v, nx_bd_v, ny_bd_v, Rf_bd_v)
        l_vec_if_1_v = compute_loss_if_1(model, u_params, x_if_v, y_if_v, Rf_if_1_v)
        l_vec_if_2_v = compute_loss_if_2(model, u_params, x_if_v, y_if_v, nx_if_v, ny_if_v, Rf_if_2_v, beta)
        L_vec_res = l_vec_res * torch.sqrt(alpha_res / torch.tensor(x_inner.size(0)))
        L_vec_bd = l_vec_bd * torch.sqrt(alpha_bd / torch.tensor(x_bd.size(0)))
        L_vec_if_1 = l_vec_if_1 * torch.sqrt(alpha_if_1 / torch.tensor(x_if.size(0)))
        L_vec_if_2 = l_vec_if_2 * torch.sqrt(alpha_if_2 / torch.tensor(x_if.size(0)))
        L_vec_res_v = l_vec_res_v / torch.sqrt(torch.tensor(x_inner_v.size(0)))
        L_vec_bd_v = l_vec_bd_v / torch.sqrt(torch.tensor(x_bd_v.size(0)))
        L_vec_if_1_v = l_vec_if_1_v / torch.sqrt(torch.tensor(x_if_v.size(0)))
        L_vec_if_2_v = l_vec_if_2_v / torch.sqrt(torch.tensor(x_if_v.size(0)))
        # print(f"L_vec_res.shape = {L_vec_res.shape}")
        # print(f"L_vec_bd.shape = {L_vec_bd.shape}")
        # print(f"L_vec_if_1.shape = {L_vec_if_1.shape}")
        # print(f"L_vec_if_2.shape = {L_vec_if_2.shape}")

        # Cat the Jacobian matrix and the residual vector
        Jac = torch.vstack([Jac_res, Jac_bd, Jac_if_1, Jac_if_2])
        L_vec = torch.vstack([L_vec_res, L_vec_bd, L_vec_if_1, L_vec_if_2])
        # print(f"Jac.shape = {Jac.shape}")
        # print(f"L_vec.shape = {L_vec.shape}")

        # Solve the linear system
        # p = qr_decomposition(Jac, L_vec, mu)
        p = cholesky(Jac, L_vec, mu)
        u_params_flatten = nn.utils.parameters_to_vector(u_params.values())
        u_params_flatten += p

        # Update the model parameters
        nn.utils.vector_to_parameters(u_params_flatten, u_params.values())

        # Compute the loss value
        loss = (
            torch.sum(L_vec_res ** 2)
            + torch.sum(L_vec_bd ** 2)
            + torch.sum(L_vec_if_1 ** 2)
            + torch.sum(L_vec_if_2 ** 2)
        )
        loss_vaild = (
            torch.sum(L_vec_res_v ** 2)
            + torch.sum(L_vec_bd_v ** 2)
            + torch.sum(L_vec_if_1_v ** 2)
            + torch.sum(L_vec_if_2_v ** 2)
        )
        savedloss.append(loss.item())
        saveloss_vaild.append(loss_vaild.item())

        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Update mu or Stop the iteration
        if loss < tol:
            # Update the model parameters
            model.load_state_dict(u_params)
            print(f"--- {time.time() - start_time:.2f} seconds ---")
            break
        elif step % 3 == 0:
            if savedloss[step] > savedloss[step - 1]:
                mu = min(mu * 2.0, 1.0e8)
            else:
                mu = max(mu / 3.0, 1.0e-10)

        if step % 200 == 0:
            # Compute the parameters alpha_bd, alpha_if_1 and alpha_if_2
            dloss_res_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_res(model, primal, x_inner, y_inner, z_inner, Rf_inner)
                )
            )(u_params)

            dloss_bd_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_bd(
                        model, primal, x_bd, y_bd, z_bd, nx_bd, ny_bd, Rf_bd
                    )
                )
            )(u_params)

            dloss_if_1_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_if_1(model, primal, x_if, y_if, Rf_if_1)
                )
            )(u_params)

            dloss_if_2_dp = grad(
                lambda primal: torch.sum(
                    compute_loss_if_2(
                        model, primal, x_if, y_if, nx_if, ny_if, Rf_if_2, beta
                    )
                )
            )(u_params)


            dloss_res_dp_flatten = nn.utils.parameters_to_vector(dloss_res_dp.values()) / torch.tensor(x_inner.size(0))
            dloss_bd_dp_flatten = nn.utils.parameters_to_vector(dloss_bd_dp.values()) / torch.tensor(x_bd.size(0))
            dloss_if_1_dp_flatten = nn.utils.parameters_to_vector(dloss_if_1_dp.values()) / torch.tensor(x_if.size(0))
            dloss_if_2_dp_flatten = nn.utils.parameters_to_vector(dloss_if_2_dp.values()) / torch.tensor(x_if.size(0))

            dloss_res_dp_norm = torch.linalg.norm(dloss_res_dp_flatten)
            dloss_bd_dp_norm = torch.linalg.norm(dloss_bd_dp_flatten)
            dloss_if_1_dp_norm = torch.linalg.norm(dloss_if_1_dp_flatten)
            dloss_if_2_dp_norm = torch.linalg.norm(dloss_if_2_dp_flatten)

            alpha_res_bar = (
                dloss_res_dp_norm
                + dloss_bd_dp_norm
                + dloss_if_1_dp_norm
                + dloss_if_2_dp_norm
            ) / dloss_res_dp_norm

            alpha_bd_bar = (
                dloss_res_dp_norm
                + dloss_bd_dp_norm
                + dloss_if_1_dp_norm
                + dloss_if_2_dp_norm
            ) / dloss_bd_dp_norm

            alpha_if_1_bar = (
                dloss_res_dp_norm
                + dloss_bd_dp_norm
                + dloss_if_1_dp_norm
                + dloss_if_2_dp_norm
            ) / dloss_if_1_dp_norm

            alpha_if_2_bar = (
                dloss_res_dp_norm
                + dloss_bd_dp_norm
                + dloss_if_1_dp_norm
                + dloss_if_2_dp_norm
            ) / dloss_if_2_dp_norm

            # Update the parameters alpha_bd, alpha_if_1 and alpha_if_2
            alpha_res = (1 - 0.1) * alpha_res + 0.1 * alpha_res_bar
            alpha_bd = (1 - 0.1) * alpha_bd + 0.1 * alpha_bd_bar
            alpha_if_1 = (1 - 0.1) * alpha_if_1 + 0.1 * alpha_if_1_bar
            alpha_if_2 = (1 - 0.1) * alpha_if_2 + 0.1 * alpha_if_2_bar
            print(f"alpha_res = {alpha_res:.2f}")
            print(f"alpha_bd = {alpha_bd:.2f}")
            print(f"alpha_if_1 = {alpha_if_1:.2f}")
            print(f"alpha_if_2 = {alpha_if_2:.2f}")


    # Save the Model
    torch.save(model, pwd + dir_name + "model_cos_6t.pt")

    # Plot the loss function
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(savedloss, "k-", label="training loss")
    ax.semilogy(saveloss_vaild, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss function over iterations")
    ax.legend()
    plt.savefig(pwd + dir_name + "loss.png", dpi=300)
    plt.show()

    # Plot the training results
    plot_x, plot_y = mesh.domain_points(50000)
    plot_x_bd, plot_y_bd = mesh.boundary_points(2000)
    plot_z = mesh.sign(plot_x, plot_y)
    plot_z_bd = torch.ones_like(plot_x_bd)
    plot_x = torch.vstack((plot_x, plot_x_bd)).to(device)
    plot_y = torch.vstack((plot_y, plot_y_bd)).to(device)
    plot_z = torch.vstack((plot_z, plot_z_bd)).to(device)
    result = functional_call(model, u_params, (plot_x, plot_y, plot_z)).cpu().detach().numpy()

    plot_theta = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
    plot_radius = mesh.func(plot_theta)
    plot_x_if = plot_radius * torch.cos(plot_theta)
    plot_y_if = plot_radius * torch.sin(plot_theta)
    plot_z_if = torch.ones_like(plot_x_if)
    plot_nx, plot_ny = mesh.compute_interface_normal_vec(plot_x_if, plot_y_if)
    plot_x_if = plot_x_if.to(device)
    plot_y_if = plot_y_if.to(device)
    plot_z_if = plot_z_if.to(device)
    plot_nx = plot_nx.to(device)
    plot_ny = plot_ny.to(device)
    pred_normal = (
        forward_dx(model, u_params, plot_x_if, plot_y_if, plot_z_if) * plot_nx +
        forward_dy(model, u_params, plot_x_if, plot_y_if, plot_z_if) * plot_ny
    ).cpu().detach().numpy()
    
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    sca = ax1.scatter(plot_x.cpu(), plot_y.cpu(), result, c=result, s=1, cmap="coolwarm")
    ax2.plot(plot_theta, pred_normal, "k-", linewidth=2)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.axes.zaxis.set_ticklabels([])
    ax1.set_title("Training Results")
    ax2.set_title("Normal Prediction")
    plt.colorbar(sca, shrink=0.5, aspect=10, pad=0.02)
    plt.savefig(pwd + dir_name + "training_results.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
