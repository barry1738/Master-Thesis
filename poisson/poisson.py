"""
Poisson equation using neural network with PyTorch.

Equation:
    Laplace(u) = f(x, y), (x, y) in [0, 1] x [0, 1]

Boundary condition:
    u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0

Exact solution:
    u(x, y) = sin(pi * x) * sin(pi * y)

Loss function:
    L = alpha * ||Laplace(u(x, p)) - f(x, y)||^2 + beta * ||u_b(x, p) - u_exact||^2


Auto-differentiation Package: PyTorch
torch.func, previously known as “functorch”, is JAX-like composable function
transforms for PyTorch.
url: https://pytorch.org/docs/stable/func.html
"""

# Import the necessary libraries
import torch
import scipy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad

# Set the default data type to double precision
torch.set_default_dtype(torch.float64)
# Set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device = ", device)


class CreateMesh:
    """Create the mesh for the Poisson equation."""
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        """Generate the interior points of the domain."""
        x = 1.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx)
        return x

    def boundary_points(self, nx):
        """Generate the boundary points of the domain."""
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
    """Define the neural network model."""
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

    def forward(self, x, y):
        """Forward pass of the model."""
        input = torch.hstack((x, y))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    
    def forward_dx(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
        return vjpfunc(torch.ones_like(output))[0]

    def forward_dy(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
        return vjpfunc(torch.ones_like(output))[0]

    def forward_dxx(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: self.forward_dx(model, params, primal, y), x)
        return vjpfunc(torch.ones_like(output))[0]

    def forward_dyy(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: self.forward_dy(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    

def weights_init(model):
    """Initialize the weights of the neural network."""
    if isinstance(model, nn.Linear):
        # nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.xavier_normal_(model.weight.data, gain=1)


def exact_sol(x, y):
    """Exact solution of the Poisson equation."""
    sol = torch.sin(1.0 * torch.pi * x) * torch.sin(1.0 * torch.pi * y)
    return sol.reshape(-1, 1)


def rhs(x, y):
    """Right-hand side of the Poisson equation."""
    f = (
        -2.0 * 1.0**2
        * torch.pi**2
        * torch.sin(1.0 * torch.pi * x)
        * torch.sin(1.0 * torch.pi * y)
    )
    return f.reshape(-1, 1)


def compute_loss_Res(model, params, x, y, Rf_inner):
    """Compute the residual loss function for the PDE residual term."""
    laplace = model.forward_dxx(model, params, x, y) + model.forward_dyy(model, params, x, y)
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
    b = torch.vstack((- diff, torch.zeros(J_mat.size(1), 1, device=device)))
    Q, R = torch.linalg.qr(A)
    x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
    return x.flatten()


def cholesky(J, diff, mu, device):
    """Solve the linear system using Cholesky decomposition"""
    A = J.t() @ J + mu * torch.eye(J.shape[1], device=device)
    b = J.t() @ -diff
    L = torch.linalg.cholesky(A)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    x = torch.linalg.solve_triangular(L.t(), y, upper=True)
    return x.flatten()

def main():
    mesh = CreateMesh()
    # interior points
    x_inner = mesh.interior_points(1000)
    x_inner_valid = mesh.interior_points(10000)
    # boundary points
    x_bd = mesh.boundary_points(100)
    x_bd_valid = mesh.boundary_points(1000)
    print(f"inner_x = {x_inner.shape}")
    print(f"boundary_x = {x_bd.shape}")

    # Convert the numpy arrays to torch tensors
    X_inner_torch = torch.from_numpy(x_inner).to(device)
    X_bd_torch = torch.from_numpy(x_bd).to(device)
    X_inner_valid_torch = torch.from_numpy(x_inner_valid).to(device)
    X_bd_valid_torch = torch.from_numpy(x_bd_valid).to(device)

    X_inner, Y_inner = X_inner_torch[:, 0].reshape(-1, 1), X_inner_torch[:, 1].reshape(-1, 1)
    X_bd, Y_bd = X_bd_torch[:, 0].reshape(-1, 1), X_bd_torch[:, 1].reshape(-1, 1)
    X_inner_valid, Y_inner_valid = X_inner_valid_torch[:, 0].reshape(-1, 1), X_inner_valid_torch[:, 1].reshape(-1, 1)
    X_bd_valid, Y_bd_valid = X_bd_valid_torch[:, 0].reshape(-1, 1), X_bd_valid_torch[:, 1].reshape(-1, 1)

    # Define the neural network model and the layers of the model, then print the model
    model = Model(2, [40], 1).to(device)  # hidden layers = [...](list), ex: [40], [10, 10] etc.
    print(model)

    # Reinitalize the weights of the model
    model.apply(weights_init)
    # Get the parameters of the model and name it as u_params
    u_params = model.state_dict()
    # for key, value in u_params.items():
    #     print(f"{key} = {value}")

    # Compute the total number of parameters
    totWb = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters = {totWb}")

    # Compute the exact solution and the right-hand side of the Poisson equation
    Rf_inner = rhs(X_inner, Y_inner)
    Rf_bd = exact_sol(X_bd, Y_bd)
    Rf_inner_valid = rhs(X_inner_valid, Y_inner_valid)
    Rf_bd_valid = exact_sol(X_bd_valid, Y_bd_valid)

    # Define the optimization parameters
    Niter = 1000
    tol = 1.0e-10
    mu = 1.0e5
    savedloss = []
    saveloss_valid = []
    alpha = 1.0
    beta = 1.0

    # Start training the model
    for step in range(Niter):
        # Compute Jacobian matrix of the loss function without the square root 
        # of the number of samples.
        # This process will get the Jacobian matrix of the loss function with
        # respect to the parameters of the model in the form of a dictionary.
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

        # Stack the Jacobian matrices into a single matrix
        Jac_res = torch.hstack([v.view(X_inner.size(0), -1) for v in jac_res_dict.values()])
        Jac_b = torch.hstack([v.view(X_bd.size(0), -1) for v in jac_b_dict.values()])

        # Divide the Jacobian matrices by the square root of the number of samples
        Jac_res = Jac_res * torch.sqrt(alpha / torch.tensor(X_inner.size(0)))
        Jac_b = Jac_b * torch.sqrt(beta / torch.tensor(X_bd.size(0)))

        # Compute the loss function without the square root of the number of samples
        L_vec_res = compute_loss_Res(model, u_params, X_inner, Y_inner, Rf_inner)
        L_vec_b = compute_loss_Bd(model, u_params, X_bd, Y_bd, Rf_bd)
        L_vec_res_valid = compute_loss_Res(model, u_params, X_inner_valid, Y_inner_valid, Rf_inner_valid)
        L_vec_b_valid = compute_loss_Bd(model, u_params, X_bd_valid, Y_bd_valid, Rf_bd_valid)

        # Divide the loss vectors by the square root of the number of samples
        L_vec_res = L_vec_res * torch.sqrt(alpha / torch.tensor(X_inner.size(0)))
        L_vec_b = L_vec_b * torch.sqrt(beta / torch.tensor(X_bd.size(0)))
        L_vec_res_valid = L_vec_res_valid / torch.sqrt(torch.tensor(X_inner_valid.size(0)))
        L_vec_b_valid = L_vec_b_valid / torch.sqrt(torch.tensor(X_bd_valid.size(0)))

        # Cat Jac_res and Jac_b into J_mat
        J_mat = torch.vstack((Jac_res, Jac_b))
        # Cat L_vec_res and L_vec_b into L_vec
        L_vec = torch.vstack((L_vec_res, L_vec_b))

        # Solve the linear system to get the update parameter
        # p = qr_decomposition(J_mat, L_vec, mu)
        p = cholesky(J_mat, L_vec, mu, device)

        # Update the parameters
        # Flatten the parameters of the model
        u_params_vec = nn.utils.parameters_to_vector(u_params.values())
        # Update the parameters of the model
        u_params_vec = u_params_vec + p
        # Convert the vector back to the parameters of the model
        nn.utils.vector_to_parameters(u_params_vec, u_params.values())

        # Compute the loss function
        loss = (
            torch.sum(L_vec_res**2) + torch.sum(L_vec_b**2)
        )
        loss_valid = (
            torch.sum(L_vec_res_valid**2) + torch.sum(L_vec_b_valid**2)
        )
        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Save the loss function
        savedloss.append(loss.item())
        saveloss_valid.append(loss_valid.item())

        if step % 50 == 0:
            # Compute the gradient of the loss function with respect to the parameters
            dloss_res_dparams = grad(
                lambda primal: torch.sum(
                    compute_loss_Res(model, primal, X_inner, Y_inner, Rf_inner)**2),
                argnums=0)(u_params)
            dloss_res_dparams_flatten = nn.utils.parameters_to_vector(dloss_res_dparams.values()) / torch.tensor(X_inner.size(0))
            dloss_res_dparams_norm = torch.linalg.norm(dloss_res_dparams_flatten)

            dloss_bd_dparams = grad(
                lambda primal: torch.sum(
                    compute_loss_Bd(model, primal, X_bd, Y_bd, Rf_bd)**2),
                argnums=0)(u_params)
            dloss_bd_dparams_flatten = nn.utils.parameters_to_vector(dloss_bd_dparams.values()) / torch.tensor(X_bd.size(0))
            d_loss_bdparams_norm = torch.linalg.norm(dloss_bd_dparams_flatten)

            # Compute the alpha_bar and beta_bar
            alpha_bar = (
                dloss_res_dparams_norm + d_loss_bdparams_norm
            ) / dloss_res_dparams_norm
            beta_bar = (
                dloss_res_dparams_norm + d_loss_bdparams_norm
            ) / d_loss_bdparams_norm

            # Update the alpha and beta
            alpha = (1-0.1) * alpha + 0.1 * alpha_bar
            beta = (1-0.1) * beta + 0.1 * beta_bar
            print(f"alpha = {alpha:.2f}, beta = {beta:.2f}")

        # If the loss function is converged, then break the loop.
        if (step == Niter - 1) or (loss < tol):
            break

        # Update the parameter mu
        if step % 3 == 0:
            if savedloss[step] > savedloss[step - 1]:
                # if the loss function is increasing, then increase the mu
                mu = min(2 * mu, 1e8)
            else:
                # if the loss function is decreasing, then decrease the mu
                mu = max(mu / 5, 1e-10)

    
    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(savedloss, "k-", label="training loss")
    ax.semilogy(saveloss_valid, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

    # Compute the exact solution
    x = torch.linspace(0, 1, 100, device=device)
    y = torch.linspace(0, 1, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    U_exact = exact_sol(X, Y).cpu()
    U_pred = functional_call(model, u_params, (X, Y)).cpu().detach().numpy()
    # U_pred = model(X, Y).cpu().detach().numpy()
    Error = torch.abs(U_exact - U_pred)

    # Plot the exact solution
    fig, axs = plt.subplots(1, 2)
    sca1 = axs[0].scatter(X.cpu(), Y.cpu(), c=U_pred)
    sca2 = axs[1].scatter(X.cpu(), Y.cpu(), c=Error)
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    axs[0].axis("square")
    axs[1].axis("square")
    plt.colorbar(sca1, shrink=0.5)
    plt.colorbar(sca2, shrink=0.5)
    plt.show()

if __name__ == "__main__":
    main()