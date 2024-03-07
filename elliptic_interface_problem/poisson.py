import torch
import scipy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device = ", device)


class CreateMesh:
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        x = 1.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx)
        return x

    def boundary_points(self, nx):
        left_x = np.hstack(
            (
                0.0 * np.ones((nx, 1)),
                1.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx),
            )
        )
        right_x = np.hstack(
            (
                np.ones((nx, 1)),
                1.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx),
            )
        )
        bottom_x = np.hstack(
            (
                1.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx),
                0.0 * np.ones((nx, 1)),
            )
        )
        top_x = np.hstack(
            (
                1.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx),
                np.ones((nx, 1)),
            )
        )
        x = np.vstack((left_x, right_x, bottom_x, top_x))
        return x


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

    def forward(self, x, y):
        input = torch.hstack((x, y))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output


def exact_sol(x, y):
    """Exact solution of the Poisson equation."""
    sol = torch.sin(2.0 * torch.pi * x) * torch.sin(2.0 * torch.pi * y)
    return sol.reshape(-1, 1)


def rhs(x, y):
    """Right-hand side of the Poisson equation."""
    f = (
        -2.0 * 2.0**2
        * torch.pi**2
        * torch.sin(2.0 * torch.pi * x)
        * torch.sin(2.0 * torch.pi * y)
    )
    return f.reshape(-1, 1)


def forward_dx(model, params, x, y):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dy(model, params, x, y):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dxx(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: forward_dx(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]

def forward_dyy(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: forward_dy(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]

def compute_loss_Res(model, params, x, y, Rf_inner):
    """Compute the residual loss function for the PDE residual term."""
    laplace = forward_dxx(model, params, x, y) + forward_dyy(model, params, x, y)
    loss_Res = laplace - Rf_inner
    return loss_Res

def compute_loss_Bd(model, params, x, y, Rf_bd):
    """Compute the boundary loss function for the PDE boundary term."""
    u_pred = functional_call(model, params, (x, y))
    loss_b = u_pred - Rf_bd
    return loss_b

def qr_decomposition(J_mat, diff, mu):
    """Solve the linear system using QR decomposition"""
    A = torch.vstack((J_mat, mu**0.5 * torch.eye(J_mat.size(1), device=device)))
    b = torch.vstack((-diff, torch.zeros(J_mat.size(1), 1, device=device)))
    Q, R = torch.linalg.qr(A)
    x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
    return x.flatten()


def main():
    mesh = CreateMesh()
    # interior points
    x_inner = mesh.interior_points(1000)
    x_inner_vaild = mesh.interior_points(10000)
    # boundary points
    x_bd = mesh.boundary_points(100)
    x_bd_vaild = mesh.boundary_points(1000)
    print(f"inner_x = {x_inner.shape}")
    print(f"boundary_x = {x_bd.shape}")

    X_inner_torch = torch.from_numpy(x_inner).to(device)
    X_bd_torch = torch.from_numpy(x_bd).to(device)
    X_inner_vaild_torch = torch.from_numpy(x_inner_vaild).to(device)
    X_bd_vaild_torch = torch.from_numpy(x_bd_vaild).to(device)

    X_inner, Y_inner = X_inner_torch[:, 0].reshape(-1, 1), X_inner_torch[:, 1].reshape(-1, 1)
    X_bd, Y_bd = X_bd_torch[:, 0].reshape(-1, 1), X_bd_torch[:, 1].reshape(-1, 1)
    X_inner_vaild, Y_inner_vaild = X_inner_vaild_torch[:, 0].reshape(-1, 1), X_inner_vaild_torch[:, 1].reshape(-1, 1)
    X_bd_vaild, Y_bd_vaild = X_bd_vaild_torch[:, 0].reshape(-1, 1), X_bd_vaild_torch[:, 1].reshape(-1, 1)

    model = Model(2, [100], 1).to(device)  # hidden layers = [...](list)
    print(model)

    # get the training parameters and total number of parameters
    u_params = dict(model.named_parameters())
    # for key, value in u_params.items():
    #     print(f"{key} = {value}")

    # 10 times the initial parameters
    u_params_flatten = nn.utils.parameters_to_vector(u_params.values()) * 10.0
    nn.utils.vector_to_parameters(u_params_flatten, u_params.values())
    print(f"Number of parameters = {u_params_flatten.numel()}")

    Rf_inner = rhs(X_inner, Y_inner)
    Rf_bd = exact_sol(X_bd, Y_bd)
    Rf_inner_vaild = rhs(X_inner_vaild, Y_inner_vaild)
    Rf_bd_vaild = exact_sol(X_bd_vaild, Y_bd_vaild)

    # Start training
    Niter = 1000
    tol = 1.0e-10
    mu = 1.0e5
    savedloss = []
    saveloss_vaild = []

    for step in range(Niter):
        # Compute Jacobian matrix
        jac_res_dict = vmap(
            jacrev(compute_loss_Res, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_inner, Y_inner, Rf_inner)

        jac_b_dict = vmap(
            jacrev(compute_loss_Bd, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_bd, Y_bd, Rf_bd)

        # Stack the Jacobian matrices
        Jac_res = torch.hstack([v.view(X_inner.size(0), -1) for v in jac_res_dict.values()])
        Jac_b = torch.hstack([v.view(X_bd.size(0), -1) for v in jac_b_dict.values()])

        # print(f"jac_res = {Jac_res.size()}")
        # print(f"jac_b = {Jac_b.size()}")
        Jac_res = Jac_res / torch.sqrt(torch.tensor(X_inner.size(0)))
        Jac_b = Jac_b / torch.sqrt(torch.tensor(X_bd.size(0)))

        # Put into loss functional to get L_vec
        L_vec_res = compute_loss_Res(model, u_params, X_inner, Y_inner, Rf_inner)
        L_vec_b = compute_loss_Bd(model, u_params, X_bd, Y_bd, Rf_bd)
        L_vec_res_vaild = compute_loss_Res(model, u_params, X_inner_vaild, Y_inner_vaild, Rf_inner_vaild)
        L_vec_b_vaild = compute_loss_Bd(model, u_params, X_bd_vaild, Y_bd_vaild, Rf_bd_vaild)
        # print(f"loss_Res = {L_vec_res.size()}")
        # print(f"loss_Bd = {L_vec_b.size()}")
        L_vec_res = L_vec_res / torch.sqrt(torch.tensor(X_inner.size(0)))
        L_vec_b = L_vec_b / torch.sqrt(torch.tensor(X_bd.size(0)))
        L_vec_res_vaild = L_vec_res_vaild / torch.sqrt(torch.tensor(X_inner_vaild.size(0)))
        L_vec_b_vaild = L_vec_b_vaild / torch.sqrt(torch.tensor(X_bd_vaild.size(0)))


        # Cat Jac_res and Jac_b into J_mat
        J_mat = torch.vstack((Jac_res, Jac_b))
        L_vec = torch.vstack((L_vec_res, L_vec_b))
        # print(f"J_mat = {J_mat.size()}")
        # print(f"L_vec = {L_vec.size()}")

        # Solve the linear system
        p = qr_decomposition(J_mat, L_vec, mu)
        # Update the parameters
        u_params_vec = nn.utils.parameters_to_vector(u_params.values())
        u_params_vec = u_params_vec + p
        nn.utils.vector_to_parameters(u_params_vec, u_params.values())

        # Compute the loss function
        loss = (
            torch.sum(L_vec_res**2) + torch.sum(L_vec_b**2)
        )
        loss_vaild = (
            torch.sum(L_vec_res_vaild**2) + torch.sum(L_vec_b_vaild**2)
        )
        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")
        savedloss.append(loss.item())
        saveloss_vaild.append(loss_vaild.item())

        if (step == Niter - 1) or (loss < tol):
            break
        
        if step % 3 == 0:
            if savedloss[step] > savedloss[step - 1]:
                mu = min(2 * mu, 1e8)
            else:
                mu = max(mu / 3, 1e-10)

    
    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(savedloss, "k-", label="training loss")
    ax.semilogy(saveloss_vaild, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

    # Update the model parameters
    model.load_state_dict(u_params)

    # Compute the exact solution
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    U_exact = exact_sol(X, Y).cpu()
    U_pred = functional_call(model, u_params, (X, Y)).cpu().detach().numpy()
    Error = torch.abs(U_exact - U_pred)

    # Plot the exact solution
    fig, axs = plt.subplots(1, 2)
    sca1 = axs[0].scatter(X.cpu(), Y.cpu(), c=U_pred)
    sca2 = axs[1].scatter(X.cpu(), Y.cpu(), c=Error)
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    axs[0].axis("square")
    axs[1].axis("square")
    plt.colorbar(sca1)
    plt.colorbar(sca2)
    plt.show()

if __name__ == "__main__":
    main()