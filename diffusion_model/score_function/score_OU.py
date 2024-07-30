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
from torch.func import functional_call, vmap, jacrev, vjp, grad


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
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

    def forward(self, x, y):
        input = torch.hstack((x, y))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

def cost_function(model, params, xin, yin, sigmaT):
    # Forward pass
    y_pred = functional_call(model, params, xin)
    # Loss
    loss = (sigmaT * yin + y_pred) / torch.sqrt(xin.shape[1])
    return loss


def main():
    # Network parameters
    model = Model(2, [40], 1).to(device)
    m = 5000  # number of training samples
    m_test = 1 * m  # number of testing samples

    # LM parameters
    max_iter = 1000  # maximum number of iterations
    tol = 1E-3  # tolerance
    opt = "Cholesky"  # "Cholesky" or "QR"
    eta = 1.0  # initial damping parameter

    # SDE parameters
    T = 10
    mu_0 = 3
    sigma_0 = 1
    beta = 3

    def mu(X, t):
        return np.exp(-beta / 2 * t) * X

    def sigma(t):
        return np.sqrt(1 - np.exp(-beta * t))

    def score_exact(X, t):
        return -(X - mu(mu_0, t)) / (np.exp(-beta * t) * sigma_0**2 + sigma(t) ** 2)

    def eps_exact(X, t):
        return -sigma(t) * score_exact(X, t)
    
    # Problem setup
    X0 = np.random.normal(mu_0, sigma_0, (m, 1))
    t = np.sort(np.append(np.random.rand(m - 1, 1), T)).reshape(1, -1)
    noise = np.random.randn(m, 1)
    Xt = mu(X0, t) + sigma(t) * noise
    sigmaT = sigma(T)
    # ... training points ...
    xin = np.vstack((X0, t))
    # ... target output ...
    yin = noise

    # Display network parameters
    print(f'Trinig points: {xin.shape[1]}')
    print(f'# of Parameters: {sum(p.numel() for p in model.parameters())}')

    # Levenberg-Marquardt algorithm
    loss = torch.zeros(max_iter)

    for step in range(max_iter):
        # ... Computation of Jacobian matrix ...
        jac_dict = vmap(
            jacrev(cost_function, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, u_params, X_inner, Y_inner, Rf_inner)


if __name__ == "__main__":
    main()
