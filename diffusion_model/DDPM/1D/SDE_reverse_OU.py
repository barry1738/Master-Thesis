"""
1D reverse Ornstein-Uhenbeck process
dXt  =
%  X(0) = X0 (const.)
%  ------------------------------
%  mean  = mu_0*exp(-beta*t)
%  var   = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
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
beta = lambda t: t
sigma = lambda t: np.sqrt(beta(t))
# ... exact mean and std ...
X_0 = 1
# ... initial condition ...
mu = lambda t: np.exp(-(t**2) / 4) * X_0
std = lambda t: np.sqrt(1 - np.exp(-(t**2) / 2))


# Ornstein-Uhenbeck process
f = lambda x, t: -beta(t) / 2 * x
g = lambda x, t: sigma(t)
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
fig, ax = plt.subplots()
ax.plot(np.linspace(0, T, M + 1), Xh_0.T)
ax.invert_xaxis()
ax.set_ylim([-10, 10])
ax.set_title('Reverse')
ax.set_xlabel('time')
ax.set_ylabel(r'$\bar{X}(t)$')
plt.autoscale(enable=True, axis="x", tight=True)
plt.show()