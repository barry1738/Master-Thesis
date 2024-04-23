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
    split_points = np.linspace(0, 1, FUNC_NUM + 1)[1:]
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


def main():
    # Define the model
    # model = DCSNNModel(in_dim=2, hidden_dim=[40], out_dim=1).to(device)

    # model = OneHotModel(
    #     in_dim=1, hidden_dim=[40], out_dim=1, part_dim=FUNC_NUM
    # ).to(device)

    model = EntityEmbeddingModel(
        in_dim=1, hidden_dim=[40], out_dim=1, part_in_dim=FUNC_NUM, part_out_dim=1
    ).to(device)

    # get the training parameters and total number of parameters
    print(f"Number of parameters = {nn.utils.parameters_to_vector(model.state_dict().values()).numel()}")

    # Create the training data and validation data
    x = qmc.LatinHypercube(d=1).random(n=200 - 2)
    x = np.vstack((0.0, x, 1.0))
    x_valid = qmc.LatinHypercube(d=1).random(n=1000)

    # z = np.argwhere(sign(x) > 0)[:, 1] + 1
    # z_valid = np.argwhere(sign(x_valid) > 0)[:, 1] + 1
    # z = sign(x)
    # z_valid = sign(x_valid)
    z = sign(x)
    z_valid = sign(x_valid)

    print(f"x = {x.shape}")
    print(f"z = {z.shape}")

    # Create the Fourier series coefficients
    rng = np.random.default_rng(10)
    a = rng.normal(loc=0.0, scale=1.0, size=(100, FUNC_NUM))
    b = rng.normal(loc=0.0, scale=5.0, size=(100, FUNC_NUM))
    c = rng.normal(loc=0.0, scale=1.0, size=(100, FUNC_NUM))
    d = rng.normal(loc=0.0, scale=5.0, size=(100, FUNC_NUM))

    # Define the right-hand side vector
    rhs_f = exact_sol(x, z, a, b, c, d)
    rhs_f_valid = exact_sol(x_valid, z_valid, a, b, c, d)

    # Plot the exact solution
    fig, ax = plt.subplots()
    ax.scatter(x_valid, rhs_f_valid, s=3, c="b")
    ax.scatter(x, rhs_f, s=3, c="r")
    plt.show()

    # Start training
    params, loss, loss_valid = networks_training(
        model=model, points_data=(x, z, x_valid, z_valid), 
        rhs_data=(rhs_f, rhs_f_valid), epochs=1000,
        tol=1.0e-12, device=device
    )
    model.load_state_dict(params)

    # print(f'embed weight = {model.ln_embed.weight}')
    # print(f"embed weight diff = {torch.diff(model.ln_embed.weight).cpu().detach().numpy()}")

    # Plot the loss function
    fig, ax = plt.subplots()
    ax.semilogy(loss, "k-", label="training loss")
    ax.semilogy(loss_valid, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss function over iterations")
    ax.legend()
    plt.show()

    # Compute the exact solution
    x_plot = np.linspace(0, 1.0, 10000).reshape(-1, 1)
    z_plot = sign(x_plot)
    x_plot_torch = torch.from_numpy(x_plot).to(device)
    z_plot_torch = torch.from_numpy(z_plot).to(device)
    U_exact = exact_sol(x_plot, z_plot, a, b, c, d)
    U_pred = functional_call(model, params, (x_plot_torch, z_plot_torch)).cpu().detach().numpy()
    Error = np.abs(U_exact - U_pred)

    # Plot the exact solution
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(x_plot, U_pred, c="r", marker=".", s=3)
    axs[1].plot(x_plot, Error, "k-")
    axs[0].set_title("Predicted solution")
    axs[1].set_title("Error")
    plt.suptitle("Function Approximation using DCSNN")
    plt.show()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = ", device)

    FUNC_NUM = 5

    main()
