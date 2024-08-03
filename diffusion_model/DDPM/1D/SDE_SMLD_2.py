"""
1D forward Langevin Dynamics process
dXt  = sqrt( d( sigma(t)^2 )/dt )*dWt
X(0) ~ N( mu_0, sigma_0^2 )
------------------------------
sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
mean = mu_0
var  = sigma_0^2 + sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# numerical setup
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
sigma_min = 0.2
sigma_max = 1
# ... initial condition ...
mu_0 = 5
sigma_0 = 0.3
X_0 = np.random.normal(mu_0, sigma_0, N)
# ... exact mean and std ...
mu_ex = mu_0
std_ex = np.sqrt(sigma_0**2 + sigma_max**2 - sigma_min**2)
# SDE setup
f = lambda x, t: 0
g = (
    lambda x, t: sigma_min
    * (sigma_max / sigma_min) ** (t / T)
    * np.sqrt(2 / T * np.log(sigma_max / sigma_min))
)
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

