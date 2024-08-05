"""
1D reverse Langevin Dynamics process
dXt  =
X(0) = X0 (const.)
------------------------------
sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
mean = X0
var  = sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
------------------------------
"""

import numpy as np


# numerical setup
d = 2  # dimension
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
sigma_min = 0.1
sigma_max = 2
# ... exact mean and std ...
X_0 = np.random.randn(d, 1)
# SDE setup
mu = lambda t: X_0
std = lambda t: np.sqrt(sigma_min**2 * ((sigma_max / sigma_min) ** (2 * t / T) - 1))
Xh_0 = np.random.multivariate_normal(mu(T).reshape(-1), std(T)**2 * np.eye(2), N).T

# Ornstein-Uhenbeck process
f = lambda x, t: 0
g = (
    lambda x, t: sigma_min
    * (sigma_max / sigma_min) ** (t / T)
    * np.sqrt(2 / T * np.log(sigma_max / sigma_min))
)
s = lambda x, t: -(x - mu(t)) / std(t)**2
eps = lambda x, t: -std(t) * s(x, t)


# Euler-Maruyama method
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        Xh_0
        + (f(Xh_0, ti) - g(Xh_0, ti) ** 2 * s(Xh_0, ti)) * (-dt)
        + g(Xh_0, ti) * np.sqrt(dt) * np.random.randn(d, N)
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