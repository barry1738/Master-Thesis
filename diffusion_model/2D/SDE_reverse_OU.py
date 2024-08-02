"""
2D forward stochastic process
dXt  = -beta*Xt*dt + sigma*dWt
X(0) = X0 ~ N(mu_0, sigma_0^2)
------------------------------
mean(t) = mu_0*exp(-beta*t)
Cov(t)  = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
score(x,t) = -(x-mean(t))/Cov(t)
-------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Numerical Setup
T = 2  # Terminal time
M = 100  # Number of iterations
dt = T / M  # Time step size
N = 2000  # Number of particles

# ... parameters in f and g ...
beta = 1
sigma = 5

# ... exact mean and std at t = 0 ...
mu_0    = np.array([[1], [2]])
sigma_0 = 2

# ... initial condition ...
mu = lambda t: mu_0 * np.exp(-beta * t)
std = lambda t: np.sqrt(
    np.exp(-2 * beta * t) * sigma_0**2
    + sigma**2 / (2 * beta) * (1 - np.exp(-2 * beta * t))
)
Xh_0 = np.random.multivariate_normal(mu(T).flatten(), std(T) ** 2 * np.eye(2), N).T
# print(f'Xh_0.shape = {Xh_0.shape}')

# Ornstein-Uhenbeck process
f = lambda x, t: -beta * x
g = lambda x, t: sigma
s = lambda x, t: -(x - mu(t)) / std(t)**2

# Euler-Maruyama method
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        Xh_0
        + (f(Xh_0, ti) - g(Xh_0, ti) ** 2 * s(Xh_0, ti)) * (-dt)
        + g(Xh_0, ti) * np.sqrt(dt) * np.random.randn(2, N)
    )

# Compute mean and covariance from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print(f"Exact.mean = \n{mu_0.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print(f"Exact.cov = \n{sigma_0**2 * np.eye(2)}")
print(f"Numer.cov = \n{cov_sde}")

xx = np.arange(mu_0[0, 0] - 5, mu_0[0, 0] + 5, 0.1)
yy = np.arange(mu_0[1, 0] - 5, mu_0[1, 0] + 5, 0.1)
X, Y = np.meshgrid(xx, yy, indexing="xy")

Z = multivariate_normal.pdf(
    np.vstack((X.flatten(), Y.flatten())).T,
    mean=mu_0.flatten(),
    cov=sigma_0 ** 2 * np.eye(2),
).reshape(X.shape)

fig, ax = plt.subplots()
ax.scatter(Xh_0[0], Xh_0[1], s=0.3, c="w")
ax.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin="lower")
plt.show()
