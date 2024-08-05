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
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta  = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... exact mean and std ...
X_0 = np.array([[2], [5]])
# SDE setup
mu = lambda t: np.exp(-(t**2) / 4) * X_0
std = lambda t: np.sqrt(1 - np.exp(-t**2 / 2))
Xh_0 = np.random.multivariate_normal(mu(T).reshape(-1), std(T)**2 * np.eye(2), N).T

# Ornstein-Uhenbeck process
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: sigma(t)
s = lambda x, t: -(x - mu(t)) / std(t)**2
eps = lambda x, t: -std(t) * s(x, t)


# Euler-Maruyama method
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        1 / np.sqrt(1 - beta(ti) * dt) * Xh_0
        - beta(ti) * dt / std(ti) * eps(Xh_0, ti)
        + np.sqrt(beta(ti) * dt) * np.random.randn(2, N)
    )

#  Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print("------------------------")
print(f"Exact.mean = \n{X_0.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print("------------------------")
print(f"Exact.cov = \n{0*np.eye(2)}")
print(f"Numer.cov = \n{cov_sde}")
print("------------------------")