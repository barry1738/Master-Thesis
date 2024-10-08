"""
Learning the 1d score function via the O-U process
dXt  = -beta/2*dXt*dt + sqrt(beta)*dWt
X(0) = X0 ~ N( mu_0, sigma_0 )
--------------------------------------
score(X,t) = -( X - mu(t) )/( exp(-beta*t)*sigma_0^2 + sigma(t)^2 )
eps(X,t)   = -sigma(t)*score(X,t)
mu(t)      = exp(-beta/2*t)*mu_0
sigma(t)   = sqrt( 1 - exp(-beta*t) )
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.func import functional_call, vmap, jacrev


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("device = ", device)


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
        # self.act = nn.LogSigmoid()

    def forward(self, x):
        input = x
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output


def weights_init(model):
    """Initialize the weights of the neural network."""
    if isinstance(model, nn.Linear):
        # nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.xavier_normal_(model.weight.data, gain=1)


def cost_function(model, params, xin, yin, sigmaT):
    loss = sigmaT * yin + functional_call(model, params, xin)
    return loss


def main():
    # Dimension
    dim = 1
    print(f"* Dimension: {dim}")

    # Network parameters
    model = Model(dim + 1, [10], dim).to(device)
    # Initialize the weights
    model.apply(weights_init)
    # Get the initial weights and biases
    wb_params = model.state_dict()

    # Training data
    m = 5000  # number of training samples
    m_test = int(1 * m)  # number of testing samples

    # SDE parameters
    T = 10.0
    mu_0 = np.arange(3.0,  3.0 + dim)
    sigma_0 = 1.0
    beta = 3.0

    def mu(X, t):
        return np.exp(-beta / 2 * t) * X

    def sigma(t):
        return np.sqrt(1 - np.exp(-beta * t))

    def score_exact(X, t):
        return -(X - mu(mu_0, t)) / (np.exp(-beta * t) * sigma_0**2 + sigma(t) ** 2)

    def eps_exact(X, t):
        return -sigma(t) * score_exact(X, t)

    # Problem setup
    # ... Generate training data (Xt, t) ...
    # ... run SDE to generate Xt ...
    # X0 = mu_0 + sigma_0 * np.random.randn(m, dim)
    X0 = np.random.normal(mu_0, sigma_0, (m, dim))
    t = np.sort(np.append(T * np.random.rand(m - 1), T)).reshape(-1, 1)
    noise = np.random.randn(m, dim)
    Xt = mu(X0, t) + sigma(t) * noise
    sigmaT = sigma(t)
    # ... training points ...
    xin = np.hstack((Xt, t))
    # ... target output ...
    yin = noise

    # Display network parameters
    totWb = sum(p.numel() for p in model.parameters())
    print(f"* Trainig points: {xin.shape[0]}")
    print(f"* # of Parameters: {totWb}")

    # Levenberg-Marquardt algorithm parameters
    max_iter = 1000  # maximum number of iterations
    tol = 1e-3  # tolerance
    opt = "Cholesky"  # "Cholesky" or "QR"
    eta = 1e1  # initial damping parameter
    loss = torch.zeros(max_iter)
    print(f"* Optimization method: {opt}")

    # Move data to device
    xin = torch.tensor(xin, device=device)
    yin = torch.tensor(yin, device=device)
    sigmaT = torch.tensor(sigmaT, device=device)
    # print(f'xin: {xin.shape}')
    # print(f'yin: {yin.shape}')
    # print(f'sigmaT: {sigmaT.shape}')

    # Training
    for epoch in range(max_iter):
        # ... Computation of Jacobian matrix ...
        jac_dict = vmap(
            jacrev(cost_function, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, wb_params, xin, yin, sigmaT)

        # Stack the Jacobian matrices
        J = -torch.hstack([v.view(m * dim, -1) for v in jac_dict.values()])
        # print(f"J.shape = {J.shape}")

        # ... Computation of the vector cost function ...
        res = cost_function(model, wb_params, xin, yin, sigmaT).reshape(-1, 1)
        # print(f"res.shape = {res.shape}")

        # ... Divide the Jacobian matrix and vector cost function by sqrt(N) ...
        J = J / torch.sqrt(torch.tensor(m * dim))
        res = res / torch.sqrt(torch.tensor(m * dim))

        # ... Update p_{k+1} using LM algorithm ...
        if opt == "Cholesky":
            # Cholesky decomposition
            L = torch.linalg.cholesky(J.T @ J + eta * torch.eye(totWb, device=device))
            x_bar = torch.linalg.solve_triangular(L, J.T @ res, upper=False)
            p = torch.linalg.solve_triangular(L.T, x_bar, upper=True)

        elif opt == "QR":
            # QR decomposition
            Q, R = torch.linalg.qr(
                torch.vstack((J, eta**0.5 * torch.eye(totWb, device=device)))
            )
            p = torch.linalg.solve_triangular(
                R,
                Q.T @ torch.vstack((res, torch.zeros(totWb, 1, device=device))),
                upper=True,
            )

        # ... Update the parameters ...
        wb_params_vec = nn.utils.parameters_to_vector(wb_params.values())
        wb_params_vec = wb_params_vec + p.flatten()
        nn.utils.vector_to_parameters(wb_params_vec, wb_params.values())

        # ... Compute the cost function ...
        res = cost_function(model, wb_params, xin, yin, sigmaT).reshape(-1, 1)
        loss[epoch] = torch.sum(res**2) / (m * dim)

        # ... Update the damping parameter ...
        if epoch % 3 == 0:
            if loss[epoch] < loss[epoch - 1]:
                eta = max(eta / 1.3, 1e-9)
            else:
                eta = min(eta * 2, 1e8)

        # ... Break the loop if the certain condition is satisfied ...
        if loss[epoch] <= tol:
            break

        # ... Display the cost function ...
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss[epoch].item():.4e}, eta: {eta:.2e}")

        # ... resampling ...
        if epoch % 10 == 0:
            # ... Generate new training data (Xt, t) ...
            # ... run SDE to generate Xt ...
            X0 = mu_0 + sigma_0 * np.random.randn(m, dim)
            t = np.sort(np.append(T * np.random.rand(m - 1), T)).reshape(-1, 1)
            noise = np.random.randn(m, dim)
            Xt = mu(X0, t) + sigma(t) * noise
            sigmaT = sigma(t)
            # ... training points ...
            xin = np.hstack((Xt, t))
            # ... target output ...
            yin = noise

            # move data to device
            xin = torch.tensor(xin, device=device)
            yin = torch.tensor(yin, device=device)
            sigmaT = torch.tensor(sigmaT, device=device)

    # Plot the loss function
    loss = loss[loss.nonzero()]  # remove zero elements
    print(f"Loss: {loss[-1].item():.4e}")
    print(f"Minimum Loss: {loss.min().item():.4e}")

    fig, ax = plt.subplots()
    ax.plot(loss.detach().numpy(), label="Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    # plt.show()

    # Testing and Output
    # ... Generate testing data (Xt, t) ...
    # ... run SDE to generate Xt ...
    X0 = mu_0 + sigma_0 * np.random.randn(m_test, dim)
    t = np.sort(np.append(T * np.random.rand(m_test - 1), T)).reshape(-1, 1)
    noise = np.random.randn(m_test, dim)
    Xt = mu(X0, t) + sigma(t) * noise
    # ... test points ...
    x_test = np.hstack((Xt, t))
    y_test = functional_call(model, wb_params, torch.tensor(x_test, device=device))
    s_test = score_exact(Xt, t)

    # Compute the 2-norm error
    # ||A||_2 = ( sum_i,j( abs(a_ij)^2 ) ) ^ (1/2)
    error_2norm = np.linalg.norm(y_test.cpu().detach().numpy() - s_test)
    print(f"Error 2-norm: {error_2norm:.6e}")


    # Plot the results
    if dim == 1:
        fig, axs = plt.subplots(1, 2)
        sca1 = axs[0].scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test.cpu().detach().numpy(),
            s=5,
            cmap="coolwarm",
        )
        axs[0].set_title(r"$score_N(X_t,t)$")
        axs[0].set_ylim([0, T])
        fig.colorbar(sca1, ax=axs[0])
        sca2 = axs[1].scatter(x_test[:, 0], x_test[:, 1], c=s_test, s=5, cmap="coolwarm")
        axs[1].set_title(r"$score(X_t,t)$")
        axs[1].set_ylim([0, T])
        fig.colorbar(sca2, ax=axs[1])
        plt.show()


if __name__ == "__main__":
    main()
