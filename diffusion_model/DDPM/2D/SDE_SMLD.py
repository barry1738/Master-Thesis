"""
1D forward Langevin Dynamics process
dXt  = sqrt( d( sigma(t)^2 )/dt )*dWt
X(0) = X0 (const.)
------------------------------
sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
mean = X0
var  = sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
------------------------------
"""

import numpy as np


# numerical setup
d = 2  # problem dimension
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
sigma_min = 0.2
sigma_max = 2
# ... initial condition ...
X_0 = np.random.rand(d, 1)
Xh_0 = np.zeros((d, N)) + X_0
# ... exact mean and std at t = T ...
mu_ex = X_0
cov_ex = (sigma_max**2 - sigma_min**2) * np.eye(d)
# SDE setup
f = lambda x, t: 0
g = (
    lambda x, t: sigma_min
    * (sigma_max / sigma_min) ** (t / T)
    * np.sqrt(2 / T * np.log(sigma_max / sigma_min))
)

# Euler-Maruyama method
for i in range(M):
    ti = i * dt
    Xh_0 = Xh_0 + f(Xh_0, ti) * dt + g(Xh_0, ti) * np.random.randn(d, N) * np.sqrt(dt)

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
