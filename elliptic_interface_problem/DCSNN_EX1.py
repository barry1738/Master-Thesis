import torch
import scipy
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, vjp, jacrev

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)


class CreateMesh:
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        x = 2.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx) - 1.0
        return x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)

    
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
        return x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
    
    def interface_points(self, nx):
        theta = 2.0 * np.pi * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx)
        x = np.hstack((
            0.2 * np.cos(theta),
            0.5 * np.sin(theta)
        ))
        return x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
    
    def sign(self, x, y):
        dist = np.sqrt((x / 0.2) ** 2 + (y / 0.5) ** 2)
        z = np.where(dist < 1.0, -1.0, 1.0)
        return z.reshape(-1, 1)
    
    def normal_vector(self, x, y):
        """
        Coompute the normal vector of interface points,
        only defined on the interface
        """
        n_x = 2.0 * x / (0.2**2)
        n_y = 2.0 * y / (0.5**2)
        length = np.sqrt(n_x**2 + n_y**2)
        normal_x = n_x / length
        normal_y = n_y / length
        return normal_x, normal_y
    

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
        self.ln_out = nn.Linear(h_dim[-1], out_dim)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, x, y, z):
        input = torch.hstack((x, y, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

def exact_sol(x, y, z):
    sol1 = torch.sin(x) * torch.sin(y)
    sol2 = torch.exp(x + y)
    sol = sol1 * (1.0 + z) / 2.0 + sol2 * (1.0 - z) / 2.0
    return sol


def rhs_f(x, y, z):
    f1 = -2.0 * torch.sin(x) * torch.sin(y)
    f2 = 2.0 * torch.exp(x + y)
    f = f1 * (1.0 + z) / 2.0 + f2 * (1.0 - z) / 2.0
    return f


def normal_u(x, y, z, nx, ny):
    u1x = torch.cos(x) * torch.sin(y)
    u1y = torch.sin(x) * torch.cos(y)
    u1 = u1x * nx + u1y * ny
    u2x = torch.exp(x + y)
    u2y = torch.exp(x + y)
    u2 = u2x * nx + u2y * ny
    normal_u = u1 * (1.0 + z) / 2.0 + u2 * (1.0 - z) / 2.0
    return normal_u


def forward_dx(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y, z)), x)
    return vjpfunc(torch.ones_like(output))[0]


def forward_dy(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal, z)), y)
    return vjpfunc(torch.ones_like(output))[0]


def forward_dxx(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: forward_dx(model, params, primal, y, z), x)
    return vjpfunc(torch.ones_like(output))[0]


def forward_dyy(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: forward_dy(model, params, x, primal, z), y)
    return vjpfunc(torch.ones_like(output))[0]


def compute_loss_Res(model, params, x, y, z, Rf_inner):
    laplace_pred = forward_dxx(model, params, x, y, z) + forward_dyy(model, params, x, y, z)
    loss_res = laplace_pred - Rf_inner
    return loss_res


def compute_loss_Bd(model, params, x, y, z, Rf_bd):
    pred = functional_call(model, params, (x, y, z))
    loss_bd = pred - Rf_bd
    return loss_bd


def compute_loss_if(model, params, x, y, Rf_if):
    z_inner = -1.0 * torch.ones_like(x)
    z_outer = 1.0 * torch.ones_like(x)
    pred_inner = functional_call(model, params, (x, y, z_inner))
    pred_outer = functional_call(model, params, (x, y, z_outer))
    pred = pred_outer - pred_inner
    loss_if = pred - Rf_if
    return loss_if

def compute_loss_if_jump(model, params, x, y, nx, ny, Rf_if_jump):
    z_inner = -1.0 * torch.ones_like(x)
    z_outer = 1.0 * torch.ones_like(x)

    dfdx_outer = forward_dx(model, params, x, y, z_outer)
    dfdy_outer = forward_dy(model, params, x, y, z_outer)
    dfdx_inner = forward_dx(model, params, x, y, z_inner)
    dfdy_inner = forward_dy(model, params, x, y, z_inner)
    normal_outer = nx * dfdx_outer + ny * dfdy_outer
    normal_inner = nx * dfdx_inner + ny * dfdy_inner
    normal_jump_pred = 1.0e-3 * normal_outer - normal_inner
    loss_normal_jump = normal_jump_pred - Rf_if_jump
    return loss_normal_jump


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
    # Create the training data and validation data
    mesh = CreateMesh()
    # interior points
    x_inner, y_inner = mesh.interior_points(1000)
    x_inner_valid, y_inner_valid = mesh.interior_points(10000)
    # boundary points
    x_bd, y_bd = mesh.boundary_points(100)
    x_bd_valid, y_bd_valid = mesh.boundary_points(1000)
    # interface points
    x_if, y_if = mesh.interface_points(500)
    x_if_valid, y_if_valid = mesh.interface_points(4000)
    print(f"inner_x = {x_inner.shape}")
    print(f"boundary_x = {x_bd.shape}")
    print(f"interface_x = {x_if.shape}")

    z_inner = mesh.sign(x_inner, y_inner)
    z_bd = mesh.sign(x_bd, y_bd)
    z_inner_valid = mesh.sign(x_inner_valid, y_inner_valid)
    z_bd_valid = mesh.sign(x_bd_valid, y_bd_valid)
    print(f"sign_z = {z_inner.shape}")
    normal_x, normal_y = mesh.normal_vector(x_if, y_if)
    normal_x_valid, normal_y_valid = mesh.normal_vector(x_if_valid, y_if_valid)
    print(f"Normal_vector = {normal_x.shape}")

    # Plot the training data
    fig, ax = plt.subplots()
    ax.scatter(x_inner, y_inner, c=z_inner, marker=".")
    ax.scatter(x_bd, y_bd, c="r", marker=".")
    ax.scatter(x_if, y_if, c="g", marker=".")
    ax.axis('square')
    plt.show()

    # Move the data to the device
    X_inner_torch = torch.from_numpy(x_inner).to(device)
    Y_inner_torch = torch.from_numpy(y_inner).to(device)
    Z_inner_torch = torch.from_numpy(z_inner).to(device)
    X_bd_torch = torch.from_numpy(x_bd).to(device)
    Y_bd_torch = torch.from_numpy(y_bd).to(device)
    Z_bd_torch = torch.from_numpy(z_bd).to(device)
    X_if_torch = torch.from_numpy(x_if).to(device)
    Y_if_torch = torch.from_numpy(y_if).to(device)
    Normal_x = torch.from_numpy(normal_x).to(device)
    Normal_y = torch.from_numpy(normal_y).to(device)

    X_inner_valid = torch.from_numpy(x_inner_valid).to(device)
    Y_inner_valid = torch.from_numpy(y_inner_valid).to(device)
    Z_inner_valid = torch.from_numpy(z_inner_valid).to(device)
    X_bd_valid = torch.from_numpy(x_bd_valid).to(device)
    Y_bd_valid = torch.from_numpy(y_bd_valid).to(device)
    Z_bd_valid = torch.from_numpy(z_bd_valid).to(device)
    X_if_valid = torch.from_numpy(x_if_valid).to(device)
    Y_if_valid = torch.from_numpy(y_if_valid).to(device)
    Normal_x_valid = torch.from_numpy(normal_x_valid).to(device)
    Normal_y_valid = torch.from_numpy(normal_y_valid).to(device)

    # Define the model
    model = Model(3, [40], 1).to(device)  # hidden layers = [...](list)
    print(model)

    # get the training parameters and total number of parameters
    u_params = dict(model.named_parameters())
    # 10 times the initial parameters
    u_params_flatten = nn.utils.parameters_to_vector(u_params.values()) * 10.0
    nn.utils.vector_to_parameters(u_params_flatten, u_params.values())
    print(f"Number of parameters = {u_params_flatten.numel()}")

    # Define the right-hand side vector
    Rf_inner = rhs_f(X_inner_torch, Y_inner_torch, Z_inner_torch)
    Rf_bd = exact_sol(X_bd_torch, Y_bd_torch, Z_bd_torch)
    Rf_if = (
        exact_sol(X_if_torch, Y_if_torch, torch.ones_like(X_if_torch)) -
        exact_sol(X_if_torch, Y_if_torch, -torch.ones_like(X_if_torch))
    )
    Rf_if_jump = (
        1.0e-3 * 
        normal_u(X_if_torch, Y_if_torch, torch.ones_like(X_if_torch), Normal_x, Normal_y) -
        normal_u(X_if_torch, Y_if_torch, -torch.ones_like(X_if_torch), Normal_x, Normal_y)
    )
    Rf_inner_valid = rhs_f(X_inner_valid, Y_inner_valid, Z_inner_valid)
    Rf_bd_valid = exact_sol(X_bd_valid, Y_bd_valid, Z_bd_valid)
    Rf_if_valid = (
        exact_sol(X_if_valid, Y_if_valid, torch.ones_like(X_if_valid)) -
        exact_sol(X_if_valid, Y_if_valid, -torch.ones_like(X_if_valid))
    )
    Rf_if_jump_valid = (
        1.0e-3 *
        normal_u(X_if_valid, Y_if_valid, torch.ones_like(X_if_valid), Normal_x_valid, Normal_y_valid) -
        normal_u(X_if_valid, Y_if_valid, -torch.ones_like(X_if_valid), Normal_x_valid, Normal_y_valid)
    )

    # Start training
    Niter = 1000
    tol = 1.0e-10
    mu = 1.0e3
    savedloss = []
    saveloss_vaild = []

    time_start = time.time()
    for step in range(Niter):
        # Compute the Jacobian matrix
        jac_res_dict = vmap(
            jacrev(compute_loss_Res, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_inner_torch, Y_inner_torch , Z_inner_torch, Rf_inner)

        jac_bd_dict = vmap(
            jacrev(compute_loss_Bd, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_bd_torch, Y_bd_torch, Z_bd_torch, Rf_bd)

        jac_if_dict = vmap(
            jacrev(compute_loss_if, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_if_torch, Y_if_torch, Rf_if)

        jac_if_jump_dict = vmap(
            jacrev(compute_loss_if_jump, argnums=1),
            in_dims=(None, None, 0, 0, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_if_torch, Y_if_torch, Normal_x, Normal_y, Rf_if_jump)

        jac_res = torch.hstack([v.view(X_inner_torch.size(0), -1) for v in jac_res_dict.values()])
        jac_bd = torch.hstack([v.view(X_bd_torch.size(0), -1) for v in jac_bd_dict.values()])
        jac_if = torch.hstack([v.view(X_if_torch.size(0), -1) for v in jac_if_dict.values()])
        jac_if_jump = torch.hstack([v.view(X_if_torch.size(0), -1) for v in jac_if_jump_dict.values()])
        # print(f"Jacobian_res = {jac_res.shape}")
        # print(f"Jacobian_bd = {jac_bd.shape}")
        # print(f"Jacobian_if = {jac_if.shape}")
        # print(f"Jacobian_if_jump = {jac_if_jump.shape}\n")

        Jac_res = jac_res / torch.sqrt(torch.tensor(jac_res.size(0)))
        Jac_bd = jac_bd / torch.sqrt(torch.tensor(jac_bd.size(0)))
        Jac_if = jac_if / torch.sqrt(torch.tensor(jac_if.size(0)))
        Jac_if_jump = jac_if_jump / torch.sqrt(torch.tensor(jac_if_jump.size(0)))

        # Compute the residual vector
        l_vec_res = compute_loss_Res(
            model, u_params, X_inner_torch, Y_inner_torch, Z_inner_torch, Rf_inner
        )
        l_vec_bd = compute_loss_Bd(
            model, u_params, X_bd_torch, Y_bd_torch, Z_bd_torch, Rf_bd
        )
        l_vec_if = compute_loss_if(
            model, u_params, X_if_torch, Y_if_torch, Rf_if
        )
        l_vec_if_jump = compute_loss_if_jump(
            model, u_params, X_if_torch, Y_if_torch, Normal_x, Normal_y, Rf_if_jump
        )
        l_vec_res_valid = compute_loss_Res(
            model, u_params, X_inner_valid, Y_inner_valid, Z_inner_valid, Rf_inner_valid
        )
        l_vec_bd_valid = compute_loss_Bd(
            model, u_params, X_bd_valid, Y_bd_valid, Z_bd_valid, Rf_bd_valid
        )
        l_vec_if_valid = compute_loss_if(
            model, u_params, X_if_valid, Y_if_valid, Rf_if_valid
        )
        l_vec_if_jump_valid = compute_loss_if_jump(
            model, u_params, X_if_valid, Y_if_valid, Normal_x_valid, Normal_y_valid, Rf_if_jump_valid
        )
        # print(f"Residual_vector = {l_vec_res.shape}")
        # print(f"Boundary_vector = {l_vec_bd.shape}")
        # print(f"Interface_vector = {l_vec_if.shape}")
        # print(f"Interface_jump_vector = {l_vec_if_jump.shape}\n")

        L_vec_res = l_vec_res / torch.sqrt(torch.tensor(l_vec_res.size(0)))
        L_vec_bd = l_vec_bd / torch.sqrt(torch.tensor(l_vec_bd.size(0)))
        L_vec_if = l_vec_if / torch.sqrt(torch.tensor(l_vec_if.size(0)))
        L_vec_if_jump = l_vec_if_jump / torch.sqrt(torch.tensor(l_vec_if_jump.size(0)))
        L_vec_res_valid = l_vec_res_valid / torch.sqrt(torch.tensor(l_vec_res_valid.size(0)))
        L_vec_bd_valid = l_vec_bd_valid / torch.sqrt(torch.tensor(l_vec_bd_valid.size(0)))
        L_vec_if_valid = l_vec_if_valid / torch.sqrt(torch.tensor(l_vec_if_valid.size(0)))
        L_vec_if_jump_valid = l_vec_if_jump_valid / torch.sqrt(torch.tensor(l_vec_if_jump_valid.size(0)))

        # Cat the Jacobian matrix and residual vector
        J_mat = torch.vstack((Jac_res, Jac_bd, Jac_if, Jac_if_jump))
        L_vec = torch.vstack((L_vec_res, L_vec_bd, L_vec_if, L_vec_if_jump))
        # print(f"Jacobian_matrix = {J_mat.shape}")
        # print(f"Residual_vector = {L_vec.shape}\n")

        # Solve the linear system
        # p = qr_decomposition(J_mat, L_vec, mu)
        p = cholesky(J_mat, L_vec, mu)
        u_params_flatten = nn.utils.parameters_to_vector(u_params.values())
        u_params_flatten += p
        # Update the model parameters
        nn.utils.vector_to_parameters(u_params_flatten, u_params.values())

        # Compute the loss function
        loss = (
            torch.sum(L_vec_res**2)
            + torch.sum(L_vec_bd**2)
            + torch.sum(L_vec_if**2)
            + torch.sum(L_vec_if_jump**2)
        )
        loss_valid = (
            torch.sum(L_vec_res_valid**2)
            + torch.sum(L_vec_bd_valid**2)
            + torch.sum(L_vec_if_valid**2)
            + torch.sum(L_vec_if_jump_valid**2)
        )
        savedloss.append(loss.item())
        saveloss_vaild.append(loss_valid.item())

        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Update mu or Stop the iteration
        if loss < tol:
            # Update the model parameters
            model.load_state_dict(u_params)
            print(f"--- {time.time() - time_start:.2f} seconds ---")
            break
        elif step % 5 == 0:
            if savedloss[step] > savedloss[step - 1]:
                mu = min(mu * 2.0, 1.0e8)
            else:
                mu = max(mu / 3.0, 1.0e-12)

    
    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(savedloss, "k-", label="training loss")
    ax.semilogy(saveloss_vaild, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss function over iterations")
    ax.legend()
    plt.show()

    # Compute the exact solution
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    Z = mesh.sign(X, Y)
    X_torch = torch.from_numpy(X).to(device)
    Y_torch = torch.from_numpy(Y).to(device)
    Z_torch = torch.from_numpy(Z).to(device)
    test_params = dict(model.named_parameters())
    U_exact = exact_sol(X_torch, Y_torch, Z_torch).cpu().detach().numpy()
    U_pred = functional_call(model, test_params, (X_torch, Y_torch, Z_torch)).cpu().detach().numpy()
    Error = np.abs(U_exact - U_pred)

    # Plot the exact solution
    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    sca1 = axs[0].scatter(X, Y, U_pred, c=U_pred, marker="o", cmap="jet")
    sca2 = axs[1].scatter(X, Y, Error, c=Error, marker="o", cmap="jet")
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    # axs[0].axis("square")
    # axs[1].axis("square")
    plt.colorbar(sca1, shrink=0.7, aspect=15, pad=0.1)
    plt.colorbar(sca2, shrink=0.7, aspect=15, pad=0.1)
    plt.suptitle('DCSNN Example 1')
    plt.show()

    # Plot the normal vector on interface
    plot_x_if, plot_y_if = mesh.interface_points(1000)
    plot_theta = np.arctan2(plot_y_if, plot_x_if)
    plot_z_if = np.ones_like(plot_x_if)
    plot_nx, plot_ny = mesh.normal_vector(plot_x_if, plot_y_if)
    plot_x_if = torch.from_numpy(plot_x_if).to(device)
    plot_y_if = torch.from_numpy(plot_y_if).to(device)
    plot_z_if = torch.from_numpy(plot_z_if).to(device)
    plot_nx = torch.from_numpy(plot_nx).to(device)
    plot_ny = torch.from_numpy(plot_ny).to(device)
    pred_normal = (
        forward_dx(model, u_params, plot_x_if, plot_y_if, plot_z_if) * plot_nx
        +
        forward_dy(model, u_params, plot_x_if, plot_y_if, plot_z_if) * plot_ny
    ).cpu().detach().numpy()
    exact_normal = (
        torch.cos(plot_x_if) * torch.sin(plot_y_if) * plot_nx
        +
        torch.sin(plot_x_if) * torch.cos(plot_y_if) * plot_ny
    ).cpu().detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(plot_theta, pred_normal, c='r', s=5)
    ax.scatter(plot_theta, exact_normal, c='k', s=1)
    plt.show()


if __name__ == "__main__":
    main()
