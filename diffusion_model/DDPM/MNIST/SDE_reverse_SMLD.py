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
d = 784  # problem dimension
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step
N = 1000  # number of particles
# ... parameters in f and g ...
sigma_min = 0.1
sigma_max = 2
# ... initial condition ...
X_0 = np.loadtxt("diffusion_model/DDPM/MNIST/X_0.csv").reshape(-1, 1)
# ... exact mean and std at t = T ...
mu = lambda t: X_0
std = lambda t: np.sqrt(sigma_min**2 * ((sigma_max / sigma_min) ** (2 * t / T) - 1))
Xh_0 = np.random.multivariate_normal(
    0 * mu(T).reshape(-1), std(T) ** 2 * np.eye(d), N
).T
# Ornstein-Uhenbeck process
f = lambda x, t: 0
g = (
    lambda x, t: sigma_min
    * (sigma_max / sigma_min) ** (t / T)
    * np.sqrt(2 / T * np.log(sigma_max / sigma_min))
)
s = lambda x, t: -(x - mu(t)) / std(t) ** 2
eps = lambda x, t: -std(t) * s(x, t)

# Euler-Maruyama method (backward)
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        Xh_0
        + (f(Xh_0, ti) - g(Xh_0, ti) ** 2.0 * s(Xh_0, ti)) * (-dt)
        + g(Xh_0, ti) * np.sqrt(dt) * np.random.randn(d, N)
    )

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
# print("------------------------")
# print(f"Exact.mean = \n{X_0}")
# print(f"Numer.mean = \n{mu_sde}")
# print("------------------------")
# print(f"Exact.cov = \n{0*np.eye(d)}")
# print(f"Numer.cov = \n{cov_sde}")
# print("------------------------")

norm = np.linalg.norm(mu_sde - X_0.flatten(), ord=np.inf)
print(f"norm of difference in mean = {norm:.6f}")
