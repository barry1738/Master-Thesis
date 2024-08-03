"""
1D forward Ornstein-Uhenbeck process
dXt  = -beta(t)/2*Xt*dt + sqrt(beta(t))*dWt
X(0) ~ N(mu_0,sigma_0^2)
------------------------------
beta = t
mean = exp(-T^2/4)*mu_0
var  = exp(-T^2/2)*sigma_0^2 + 1 - exp(-T^2/2)
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# numerical setup
T = 1  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... initial condition ...
mu_0 = 5
sigma_0 = 0.3
X_0 = np.random.normal(mu_0, sigma_0, N)
# ... exact mean and std ...
mu_ex = np.exp(-(T**2) / 4) * mu_0
std_ex = np.sqrt(np.exp(-(T**2) / 2) * sigma_0**2 + 1 - np.exp(-(T**2) / 2))
# SDE setup
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: sigma(t)

# Euler-Maruyama method
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, 0] = X_0

for i in range(M):
    ti = i * dt
    Xh_0[:, i + 1] = (
        Xh_0[:, i]
        + f(Xh_0[:, i], ti) * dt
        + g(Xh_0[:, i], ti) * np.sqrt(dt) * np.random.randn(N)
    )

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, -1]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, -1] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact.mean: {mu_ex:.6f}")
print(f'Numer.mean: {mu_sde:.6f}')
print("--------------------")
print(f"Exact.std : {std_ex:.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].set_ylim([-10, 10])
axs[0].set_xlabel("time")
axs[0].set_ylabel("X(t)")
axs[0].set_title("Reverse")

axs[1].plot(-0.1 * np.ones_like(Xh_0[:, -1]), Xh_0[:, -1], "k.")
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_ex, std_ex), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()

