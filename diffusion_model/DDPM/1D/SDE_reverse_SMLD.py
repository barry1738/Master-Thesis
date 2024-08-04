"""
1D reverse Ornstein-Uhenbeck process
dXt  =
X(0) = N( mu_0, sigma_0^2 )
------------------------------
mean  = mu_0*exp(-beta*t)
var   = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
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
# ... parameters in f and g ...
sigma_min = 0.1
sigma_max = 2
# ... exact mean and std ...
X_0 = 3
# ... initial condition ...
mu = lambda t: X_0
std = lambda t: np.sqrt(sigma_min**2 * ((sigma_max / sigma_min) ** (2 * t / T) - 1))

# Ornstein-Uhenbeck process
f = lambda x, t: 0
g = (
    lambda x, t: sigma_min
    * (sigma_max / sigma_min) ** (t / T)
    * np.sqrt(2 / T * np.log(sigma_max / sigma_min))
)
s = lambda x, t: - (x - mu(t)) / std(t) ** 2
eps = lambda x, t: -std(t) * s(x, t)

# Euler-Maruyama method (backward)
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, -1] = np.random.normal(mu(T), std(T), N)

for i in range(M, 0, -1):
    ti = i * dt
    Xh_0[:, i - 1] = (
        Xh_0[:, i]
        + (f(Xh_0[:, i], ti) - g(Xh_0[:, i], ti) ** 2 * s(Xh_0[:, i], ti)) * (-dt)
        + g(Xh_0[:, i], ti) * np.sqrt(dt) * np.random.randn(N)
    )

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, 0]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, 0] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact mean: {X_0:.6f}")
print(f"Numer.mean: {mu_sde:.6f}")
print("--------------------")
print(f"Exact std : {std(0):.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 1)
axs.plot(np.linspace(0, T, M + 1), Xh_0.T)
axs.invert_xaxis()
axs.set_ylim([-10, 10])
axs.set_title('Reverse')
axs.set_xlabel('time')
axs.set_ylabel(r'$\bar{X}(t)$')
axs.autoscale(enable=True, axis="x", tight=True)
# axs[1].scatter(-0.1 * np.ones_like(Xh_0[:, 0]), Xh_0[:, 0], c="k", s=1)
# axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_0, std(0)), np.arange(-10, 10, 0.1))
# axs[1].set_axis_off()
plt.show()