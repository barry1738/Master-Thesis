import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import qmc
from torch.func import functional_call
from networks_training import networks_training


class DCSNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DCSNNModel, self).__init__()
        # activation function
        self.act = nn.Sigmoid()

        # input layer
        self.ln_in = nn.Linear(in_dim, hidden_dim[0])

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(hidden_dim[-1], out_dim, bias=True)

    def forward(self, x, z):
        input = torch.hstack((x, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output


class OneHotModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, part_dim):
        super(OneHotModel, self).__init__()
        # activation function
        self.act = nn.Sigmoid()

        # input layer
        self.ln_in = nn.Linear(in_dim + part_dim, hidden_dim[0])

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(hidden_dim[-1], out_dim, bias=True)

    def forward(self, x, z):
        input = torch.hstack((x, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output


class EntityEmbeddingModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, part_in_dim, part_out_dim):
        super(EntityEmbeddingModel, self).__init__()
        # activation function
        self.act = nn.Sigmoid()

        # input layer
        self.ln_in = nn.Linear(in_dim + part_out_dim, hidden_dim[0])

        # hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(hidden_dim[-1], out_dim, bias=True)

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
    

def sign(x):
    split_points = np.linspace(Xmin, Xmax, FUNC_NUM + 1)[1:]
    comparison = (x <= split_points).astype(int)
    z_values = np.zeros((len(x), FUNC_NUM), dtype=np.float64)
    z_values[np.arange(len(x)), np.argmax(comparison, axis=1)] = 1
    return z_values


def exact_sol(x, z, a, b, c, d):
    """Exact solution of the function."""
    # Fourier series functions
    four_func = np.array([
        np.sum(a[:, i] * np.cos(b[:, i] * x) + c[:, i] * np.sin(d[:, i] * x), axis=1)
        for i in range(FUNC_NUM)
    ])
    four_func = np.array(four_func).T
    sol = np.sum(z * four_func, axis=1)
    return sol.reshape(-1, 1)


def loss_figure(loss, loss_valid, step):
    """Plot the loss function over iterations."""
    fig, ax = plt.subplots()
    ax.semilogy(loss, "k-", label="training loss")
    ax.semilogy(loss_valid, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss function over iterations")
    ax.legend()
    # plt.show()
    plt.savefig(dir + f"\\loss_function_{step}.png", dpi=300)


def result_figure(x, U_pred, Error, step):
    """Plot the exact solution and predicted solution."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(x, U_pred, c="r", marker=".", s=3)
    axs[1].plot(x, Error, "k-")
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    # plt.show()
    plt.savefig(dir + f"\\result_{step}.png", dpi=300)


def main():
    print(f"Training TYPE = {TYPE}")

    # Define the model
    if TYPE == "DCSNN":
        model = DCSNNModel(in_dim=2, hidden_dim=[50], out_dim=1).to(device)
    elif TYPE == "OneHot":
        model = OneHotModel(
            in_dim=1, hidden_dim=[50], out_dim=1, part_dim=FUNC_NUM
        ).to(device)
    elif TYPE == "EntityEmbedding":
        model = EntityEmbeddingModel(
            in_dim=1,
            hidden_dim=[50],
            out_dim=1,
            part_in_dim=FUNC_NUM,
            part_out_dim=10,
        ).to(device)

    # get the training parameters and total number of parameters
    print(f"Number of parameters = {nn.utils.parameters_to_vector(model.state_dict().values()).numel()}")

    # Create the training data and validation data
    x = np.vstack((Xmin, Xmax * qmc.LatinHypercube(d=1).random(n=200 - 2), Xmax))
    x_valid = Xmax * qmc.LatinHypercube(d=1).random(n=10000)
    x_plot = np.linspace(Xmin, Xmax, 10000).reshape(-1, 1)

    z_matrix = sign(x)
    z_matrix_valid = sign(x_valid)
    z_matrix_plot = sign(x_plot)

    if TYPE == "DCSNN":
        z = (np.argwhere(z_matrix > 0)[:, 1] + 1).reshape(-1, 1)
        z_valid = (np.argwhere(z_matrix_valid > 0)[:, 1] + 1).reshape(-1, 1)
        z_plot = (np.argwhere(z_matrix_plot > 0)[:, 1] + 1).reshape(-1, 1)
    elif TYPE == "OneHot":
        z = z_matrix.copy()
        z_valid = z_matrix_valid.copy()
        z_plot = z_matrix_plot.copy()
    elif TYPE == "EntityEmbedding":
        z = z_matrix.copy()
        z_valid = z_matrix_valid.copy()
        z_plot = z_matrix_plot.copy()

    # Create the Fourier series coefficients
    rng = np.random.default_rng(10)
    a = rng.normal(loc=0.0, scale=1.0, size=(100, FUNC_NUM))
    b = rng.normal(loc=0.0, scale=5.0, size=(100, FUNC_NUM))
    c = rng.normal(loc=0.0, scale=1.0, size=(100, FUNC_NUM))
    d = rng.normal(loc=0.0, scale=5.0, size=(100, FUNC_NUM))

    # Define the right-hand side vector
    rhs_f = exact_sol(x, z_matrix, a, b, c, d)
    rhs_f_valid = exact_sol(x_valid, z_matrix_valid, a, b, c, d)

    # Convert the data to torch tensors
    x_plot_torch = torch.from_numpy(x_plot).to(device)
    z_plot_torch = torch.from_numpy(z_plot).to(device)
    U_exact = exact_sol(x_plot, z_matrix_plot, a, b, c, d)

    # Plot the exact solution
    # fig, ax = plt.subplots()
    # ax.scatter(x_valid, rhs_f_valid, s=3, c="b")
    # ax.scatter(x, rhs_f, s=3, c="r")
    # plt.show()

    Inf_norm = []
    L2_norm = []
    Success_training = 0

    # Start training
    while Success_training < 5:
        print(f"step = {Success_training}")
        params, loss, loss_valid = networks_training(
            model=model, points_data=(x, z, x_valid, z_valid), 
            rhs_data=(rhs_f, rhs_f_valid), epochs=5000,
            tol=1.0e-12, device=device
        )
        # model.load_state_dict(params)

        if loss[-1] < 1.0e0:
            # Compute infinity and L2 norm of the error
            U_pred = functional_call(model, params, (x_plot_torch, z_plot_torch)).cpu().detach().numpy()
            Error = np.abs(U_exact - U_pred)
            Inf_norm.append(np.max(Error))
            L2_norm.append(np.sqrt(np.sum(Error**2) / len(Error)))

            Success_training += 1

            # Plot the loss function
            loss_figure(loss, loss_valid, Success_training)
            # Plot the result
            result_figure(x_plot, U_pred, Error, Success_training)


    # if TYPE == "EntityEmbedding":
    #     print(f'embed weight = {model.ln_embed.weight}')
    #     print(f"embed weight diff = {torch.diff(model.ln_embed.weight).cpu().detach().numpy()}")

    # print the average infinity and L2 norm of the error
    avg_inf_norm = np.mean(Inf_norm)
    avg_l2_norm = np.mean(L2_norm)
    print("**************************************************")
    print(f"inf_norm = {avg_inf_norm:.2e}, l2_norm = {avg_l2_norm:.2e}")
    print("**************************************************")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = ", device)

    plt.rcParams.update({"font.size": 12})

    FUNC_NUM = 10

    # TYPE = "DCSNN"
    # TYPE = "OneHot"
    TYPE = "EntityEmbedding"

    Xmin = 0.0
    Xmax = 1.0
    # Xmax = 2 * np.pi

    dir = "C:\\Users\\barry\\Desktop\\" + TYPE
    if not os.path.exists(dir):
        print("Creating data directory...")
        os.makedirs(dir)

    main()
