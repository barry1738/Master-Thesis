"""
1D reverse Ornstein-Uhenbeck process
dXt  =
X(0) = N( mu_0, sigma_0^2 )
------------------------------
mean  = 
var   = 
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# numerical setup
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... exact mean and std ...
mu_0 = 3
sigma_0 = 1
# ... initial condition ...
mu = lambda t: np.exp(-(t**2) / 4) * mu_0
std = lambda t: np.sqrt(np.exp(-(t**2) / 2) * sigma_0**2 + 1 - np.exp(-(t**2) / 2))

# reverse SDE process
beta = lambda t: t
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: np.sqrt(beta(t))
s = lambda x, t: - (x - mu(t)) / std(t) ** 2
sigma = lambda t: np.sqrt(1 - np.exp(-(t**2) / 2))
eps = lambda x, t: -sigma(t) * s(x, t)

# Euler-Maruyama method (backward)
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, -1] = np.random.normal(mu(T), std(T), N)

for i in range(M, 0, -1):
    ti = i * dt
    Xh_0[:, i - 1] = (
        (1 + beta(ti) * dt / 2) * Xh_0[:, i]
        -beta(ti) * dt / sigma(ti) * eps(Xh_0[:, i], ti)
        + np.sqrt(beta(ti) * dt) * np.random.randn(N)
    )

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, 0]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, 0] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact mean: {mu_0:.6f}")
print(f"Numer.mean: {mu_sde:.6f}")
print("--------------------")
print(f"Exact std : {std(0):.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2)
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].invert_xaxis()
axs[0].set_ylim([-10, 10])
axs[0].set_title('Reverse')
axs[0].set_xlabel('time')
axs[0].set_ylabel(r'$\bar{X}(t)$')
axs[0].autoscale(enable=True, axis="x", tight=True)
axs[1].scatter(-0.1 * np.ones_like(Xh_0[:, 0]), Xh_0[:, 0], c="k", s=1)
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_0, std(0)), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()