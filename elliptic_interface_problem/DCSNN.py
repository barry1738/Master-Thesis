import torch
import time
import numpy as np
import torch.nn as nn
import scipy.stats.qmc as qmc
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, vjp, jacrev


torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)


class CreateMesh:
    def __init__(self) -> None:
        pass

    def domain_points(self, n, *, xc=0, yc=0, r=1):
        """Uniform random distribution within a circle"""
        radius = r * np.sqrt(qmc.LatinHypercube(d=1).random(n=n))
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)
    
    def boundary_points(self, n, *, xc=0, yc=0, r=1):
        """Uniform random distribution on a circle"""
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        x = xc + r * np.cos(theta)
        y = yc + r * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)

    def interface_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a polar curve"""
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        radius = 1 / 2 + np.cos(3 * theta) / 10
        # radius = 0.5
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)

    def sign(self, x, y):
        """Check the points inside the polar curve, return z = -1 if inside, z = 1 if outside"""
        dist = torch.sqrt(x ** 2 + y ** 2) - (1 / 2 + torch.cos(3 * torch.atan2(y, x)) / 10)
        # dist = torch.sqrt(x ** 2 + y ** 2) - 0.5
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
        ri = 1 / 2 + torch.cos(3 * theta) / 10
        dri = -3 * torch.sin(3 * theta) / 10
        # ri = 0.5
        # dri = 0
        nx = dri * torch.sin(theta) + ri * torch.cos(theta)
        ny = -dri * torch.cos(theta) + ri * torch.sin(theta)
        dist = torch.sqrt(nx ** 2 + ny ** 2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny
    
    def compute_interface_curvature(self, x, y):
        """Compute the interface curvature"""
        theta = torch.atan2(y, x)
        r = 1 / 2 + torch.cos(3 * theta) / 10
        drdt = -3 * torch.sin(3 * theta) / 10
        d2rdt2 = -9 * torch.cos(3 * theta) / 10
        # r = 0.5
        # drdt = 0
        # d2rdt2 = 0
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
        self.ln_out = nn.Linear(h_dim[-1], out_dim, bias=False)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()

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
    pred = pred_outer - beta * pred_inner
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
    # Create the training data
    mesh = CreateMesh()
    x_inner, y_inner = mesh.domain_points(1000)
    x_bd, y_bd = mesh.boundary_points(200)
    x_if, y_if = mesh.interface_points(1000)
    z_inner = mesh.sign(x_inner, y_inner)
    z_bd = torch.ones_like(x_bd)

    # Create the validation data
    x_inner_v, y_inner_v = mesh.domain_points(10000)
    x_bd_v, y_bd_v = mesh.boundary_points(1000)
    x_if_v, y_if_v = mesh.interface_points(2000)
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
    fig, ax = plt.subplots()
    ax.scatter(x_inner, y_inner, c=z_inner, s=1)
    ax.scatter(x_bd, y_bd, s=1)
    sca = ax.scatter(x_if, y_if, c=k_if, s=1)
    # ax.quiver(x_bd, y_bd, nx_bd, ny_bd, angles="xy", scale_units="xy", scale=5)
    # ax.quiver(x_if, y_if, nx_if, ny_if, angles="xy", scale_units="xy", scale=5)
    ax.axis('equal')
    plt.colorbar(sca)
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

    # Define the model
    model = Model(3, [50, 50], 1).to(device)
    # print(model)

    # get the training parameters and total number of parameters
    u_params = dict(model.named_parameters())
    # 10 times the initial parameters
    u_params_flatten = nn.utils.parameters_to_vector(u_params.values()) * 10.0
    nn.utils.vector_to_parameters(u_params_flatten, u_params.values())
    print(f"Number of parameters = {u_params_flatten.numel()}")

    # Define the right-hand side vector
    Ca = torch.tensor(100)
    beta = torch.tensor(100)
    Rf_inner = torch.zeros_like(x_inner)
    Rf_bd = torch.zeros_like(x_bd)
    Rf_if_1 = k_if / Ca - (1 - 1 / beta) * torch.log(r_if) / (2 * torch.pi)
    # Rf_if_1 = k_if / Ca
    Rf_if_2 = torch.zeros_like(x_if)

    Rf_inner_v = torch.zeros_like(x_inner_v)
    Rf_bd_v = torch.zeros_like(x_bd_v)
    Rf_if_1_v = k_if_v / Ca - (1 - 1 / beta) * torch.log(r_if_v) / (2 * torch.pi)
    # Rf_if_1_v = k_if_v / Ca
    Rf_if_2_v = torch.zeros_like(x_if_v)

    print(f"Rf_if_1 = {torch.max(Rf_if_1)}")


    # Start the training
    Niter = 1000
    tol = 1.0e-9
    mu = 1.0e3
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

        Jac_res = jac_res / torch.sqrt(torch.tensor(x_inner.size(0)))
        Jac_bd = jac_bd / torch.sqrt(torch.tensor(x_bd.size(0)))
        Jac_if_1 = jac_if_1 / torch.sqrt(torch.tensor(x_if.size(0)))
        Jac_if_2 = jac_if_2 / torch.sqrt(torch.tensor(x_if.size(0)))
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
        L_vec_res = l_vec_res / torch.sqrt(torch.tensor(x_inner.size(0)))
        L_vec_bd = l_vec_bd / torch.sqrt(torch.tensor(x_bd.size(0)))
        L_vec_if_1 = l_vec_if_1 / torch.sqrt(torch.tensor(x_if.size(0)))
        L_vec_if_2 = l_vec_if_2 / torch.sqrt(torch.tensor(x_if.size(0)))
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
        p = qr_decomposition(Jac, L_vec, mu)
        # p = cholesky(Jac, L_vec, mu)
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
                mu = max(mu / 3, 1.0e-10)


    # Save the Model
    torch.save(model.state_dict(), "model.pt")

    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(savedloss, "k-", label="training loss")
    ax.semilogy(saveloss_vaild, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss function over iterations")
    ax.legend()
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

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(plot_x.to('cpu'), plot_y.to('cpu'), result, c=result, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


if __name__ == '__main__':
    main()
