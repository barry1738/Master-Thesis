import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import qmc
from torch.func import functional_call, vmap, jacrev

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)
    

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, part_in_dim, part_out_dim):
        super(Model, self).__init__()
        # activation function
        self.act = nn.Sigmoid()

        # input layer
        self.ln_in = nn.Linear(in_dim + part_out_dim, hidden_dim[0])

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(hidden_dim[-1], out_dim, bias=True)  # bias=True or False?

        # embed layer
        self.ln_embed = nn.Linear(part_in_dim, part_out_dim, bias=False)


    def forward(self, x, z):
        z = self.ln_embed(z)
        input = torch.hstack((x, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

def weights_init(model):
    """Initialize the weights of the neural network."""
    if isinstance(model, nn.Linear):
        # nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.xavier_normal_(model.weight.data, gain=2)


def exact_sol(x, z):
    sol1 = np.sin(2 * np.pi * x).reshape(-1)
    sol2 = np.cos(2 * np.pi * x).reshape(-1)
    sol = sol1 * z[:, 0] + sol2 * z[:, 1]
    return sol.reshape(-1, 1)


def compute_loss(model, params, x, z, rhs_f):
    loss = functional_call(model, params, (x, z)) - rhs_f
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
    # Create the training data and validation data
    x = qmc.LatinHypercube(d=1).random(n=100-2)
    x_valid = qmc.LatinHypercube(d=1).random(n=1000)
    x = np.vstack((0.0, x, 1.0))
    # if x < 0.5: z = [1, 0], else: z = [0, 1]
    z = np.where(x < 0.5, [1.0, 0.0], [0.0, 1.0])
    z_valid = np.where(x_valid < 0.5, [1.0, 0.0], [0.0, 1.0])
    print(f"x = {x.shape}")
    print(f"z = {z.shape}")

    # Plot the exact solution
    y = exact_sol(x, z)
    print(f"y = {y.shape}")

    # fig, ax = plt.subplots()
    # ax.scatter(x, y, s=1)
    # plt.show()

    # Move the data to the device
    x_torch = torch.from_numpy(x).to(device)
    z_torch = torch.from_numpy(z).to(device)
    x_valid_torch = torch.from_numpy(x_valid).to(device)
    z_valid_torch = torch.from_numpy(z_valid).to(device)

    # Define the model
    model = Model(
        in_dim=1, hidden_dim=[40], out_dim=1, part_in_dim=2, part_out_dim=2
    ).to(device)
    model.apply(weights_init)
    print(model)

    # get the training parameters and total number of parameters
    u_params = model.state_dict()
    # # 10 times the initial parameters
    u_params_flatten = nn.utils.parameters_to_vector(u_params.values())
    print(f"Number of parameters = {u_params_flatten.numel()}")

    # predict = functional_call(model, u_params, (x_torch, z_torch))
    # print(f'predict = {predict.shape}')

    # Define the right-hand side vector
    rhs_f = torch.from_numpy(exact_sol(x, z)).to(device)
    rhs_f_valid = torch.from_numpy(exact_sol(x_valid, z_valid)).to(device)

    # Start training
    Niter = 1000
    tol = 1.0e-10
    mu = 1.0e3
    savedloss = []
    saveloss_vaild = []

    ts = time.time()
    for step in range(Niter):
        # Compute the Jacobian matrix
        jac_dict = vmap(
            jacrev(compute_loss, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, x_torch, z_torch, rhs_f)

        jac = torch.hstack([v.view(x_torch.size(0), -1) for v in jac_dict.values()])
        jac /= torch.sqrt(torch.tensor(x_torch.size(0)))
        # print(f"Jacobian_res = {jac.shape}")

        # Compute the residual vector
        l_vec = compute_loss(model, u_params, x_torch, z_torch, rhs_f)
        l_vec_valid = compute_loss(
            model, u_params, x_valid_torch, z_valid_torch, rhs_f_valid)
        # print(f"Residual_vector = {l_vec.shape}")

        l_vec /= torch.sqrt(torch.tensor(x_torch.size(0)))
        l_vec_valid /= torch.sqrt(torch.tensor(x_valid_torch.size(0)))

        # print(f"Jacobian_matrix shape = {jac.shape}")
        # print(f"Residual_vector shape = {l_vec.shape}\n")

        # Solve the linear system
        # p = qr_decomposition(J_mat, L_vec, mu)
        p = cholesky(jac, l_vec, mu)
        u_params_flatten = nn.utils.parameters_to_vector(u_params.values())
        u_params_flatten += p
        # Update the model parameters
        nn.utils.vector_to_parameters(u_params_flatten, u_params.values())

        # Compute the loss function
        loss = torch.sum(l_vec**2)
        loss_valid = torch.sum(l_vec_valid**2)
        savedloss.append(loss.item())
        saveloss_vaild.append(loss_valid.item())

        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Update mu or Stop the iteration
        if loss < tol:
            # Update the model parameters
            model.load_state_dict(u_params)
            print(f"--- {time.time() - ts:.4f} seconds ---")
            break
        elif step % 3 == 0:
            if savedloss[step] > savedloss[step - 1]:
                mu = min(mu * 2.0, 1.0e8)
            else:
                mu = max(mu / 5.0, 1.0e-10)

    
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
    x_plot = np.linspace(0, 1, 1000).reshape(-1, 1)
    z_plot = np.where(x_plot < 0.5, [1.0, 0.0], [0.0, 1.0])
    x_plot_torch = torch.from_numpy(x_plot).to(device)
    z_plot_torch = torch.from_numpy(z_plot).to(device)
    U_exact = exact_sol(x_plot, z_plot)
    U_pred = functional_call(model, u_params, (x_plot_torch, z_plot_torch)).cpu().detach().numpy()
    Error = np.abs(U_exact - U_pred)

    # Plot the exact solution
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(x_plot, U_pred, c='r', marker=".", s=3)
    axs[1].plot(x_plot, Error, 'k-')
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    plt.suptitle('Function Approximation using DCSNN')
    plt.show()


if __name__ == "__main__":
    main()
