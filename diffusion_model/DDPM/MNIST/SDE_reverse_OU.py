"""
2D reverse Ornstein-Uhenbeck process
dXt  =
X(0) = X0 (const.)
------------------------------
mean  = mu_0*exp(-beta*t)
var   = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
------------------------------
"""

import numpy as np


# numerical setup
d = 784  # problem dimension
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step
N = 1000  # number of particles
# ... parameters in f and g ...
beta = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... initial condition ...
X_0 = np.loadtxt("diffusion_model/DDPM/MNIST/X_0.csv").reshape(-1, 1)
# ... exact mean and std at t = T ...
mu = lambda t: np.zeros((d, 1))
std = lambda t: 1
Xh_0 = np.random.multivariate_normal(mu(T).reshape(-1), std(T) ** 2 * np.eye(d), N).T
# Ornstein-Uhenbeck process
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: sigma(t)
mu = lambda t: np.exp(-(t ** 2) / 4) * X_0
std = lambda t: np.sqrt(1 - np.exp(-t ** 2 / 2))
s = lambda x, t: -(x - mu(t)) / std(t) ** 2
eps = lambda x, t: -std(t) * s(x, t)

# Euler-Maruyama method (backward)
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        Xh_0
        + (f(Xh_0, ti) - g(Xh_0, ti) ** 2.0 * s(Xh_0, ti)) * (-dt)
        + g(Xh_0, ti) * np.sqrt(dt) * np.random.randn(d, N)
    )

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
# print("------------------------")
# print(f"Exact.mean = \n{X_0}")
# print(f"Numer.mean = \n{mu_sde}")
# print("------------------------")
# print(f"Exact.cov = \n{0*np.eye(d)}")
# print(f"Numer.cov = \n{cov_sde}")
# print("------------------------")

norm = np.linalg.norm(mu_sde - X_0.flatten(), ord=np.inf)
print(f"norm of difference in mean = {norm:.6f}")
