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
d = 784  # problem dimension
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step
N = 2000  # number of particles
# ... parameters in f and g ...
beta = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... initial condition ...
X_0 = np.loadtxt('diffusion_model/DDPM/MNIST/X_0.csv').reshape(-1, 1)
Xh_0 = np.zeros((d, N)) + X_0
# ... exact mean and std at t = T ...
mu_ex = np.exp(-(T**2) / 4) * X_0
cov_ex = (1 - np.exp(-T**2 / 2)) * np.eye(d)
# SDE setup
f = lambda t, x: -beta(t) / 2 * x
g = lambda t, x: sigma(t)

# Euler-Maruyama method
for i in range(M):
    ti = i * dt
    Xh_0 = Xh_0 + f(ti, Xh_0) * dt + g(ti, Xh_0) * np.random.randn(d, N) * np.sqrt(dt)

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print("------------------------")
print(f"Exact.mean = \n{mu_ex}")
print(f"Numer.mean = \n{mu_sde}")
print("------------------------")
print(f"Exact.cov = \n{cov_ex}")
print(f"Numer.cov = \n{cov_sde}")
print("------------------------")
