"""
2D forward Ornstein-Uhenbeck process
dXt  = -beta(t)/2*Xt*dt + sqrt(beta(t))*dWt
X(0) = X0 (const.)
------------------------------
mean = exp(-T^2/4)*X0
var  = 1 - exp(-T^2/2)
------------------------------
"""

import numpy as np


# numerical setup
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... initial condition ...
X0 = np.array([[2], [3]])
Xh_0 = np.zeros((2, N)) + X0
# ... exact mean and std at t = T ...
mu_ex = np.exp(-T**2 / 4) * X0
cov_ex = (1 - np.exp(-T**2 / 2)) * np.eye(2)
# SDE setup
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: sigma(t)

# Euler-Maruyama method
for i in range(M):
    ti = i * dt
    Xh_0 = Xh_0 + f(Xh_0, ti) * dt + g(Xh_0, ti) * np.random.randn(2, N) * np.sqrt(dt)

#  Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print("------------------------")
print(f"Exact.mean = \n{mu_ex.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print("------------------------")
print(f"Exact.cov = \n{cov_ex}")
print(f"Numer.cov = \n{cov_sde}")
print("------------------------")
