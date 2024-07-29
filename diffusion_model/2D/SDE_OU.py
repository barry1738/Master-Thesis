"""
2D forward stochastic process
dXt  = F*Xt*dt + G*dWt
X(0) = X0 ~ N(mu_0,SIGMA_0)
-------------------------
mean = exp(t*F)*mu_0
Cov  = solved by RK4
-------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import expm
from RK4_OU_covariance import RK4_OU


# Numerical Setup
T = 2  # Terminal time
M = 100  # Number of iterations
dt = T / M  # Time step size
N = 2000  # Number of particles

# ... construct f, SIGMA and SIGMA^(1/2) ...
F = np.random.randn(2, 2)
F = - F @ F.T
G = np.random.randn(2, 2)
G = G @ G.T

# ... initial condition ...
mu_0 = np.array([[1], [2]])
SIGMA_0 = np.random.randn(2, 2)
SIGMA_0 = SIGMA_0.T @ SIGMA_0
Xh_0 = np.random.multivariate_normal(mu_0.flatten(), SIGMA_0, N).T

# ... exact mean and covariance at t = T ...
mu_ex = expm(T * F) @ mu_0
cov_ex = RK4_OU(T, F, G, SIGMA_0)

# SDE setup
f = lambda x, t: F @ x
g = lambda x, t: G

# Euler-Maruyama method
for i in range(M):
    ti = i * dt
    Xh_0 = Xh_0 + f(Xh_0, ti) * dt + g(Xh_0, ti) @ np.random.randn(2, N) * np.sqrt(dt)

# Compute mean and covariance from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print(f"Exact.mean = \n{mu_ex.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print(f"Exact.cov = \n{cov_ex}")
print(f"Numer.cov = \n{cov_sde}")

xx = np.arange(mu_sde[0] - 5, mu_sde[0] + 5, 0.1)
yy = np.arange(mu_sde[1] - 5, mu_sde[1] + 5, 0.1)
X, Y = np.meshgrid(xx, yy, indexing="xy")

Z = multivariate_normal.pdf(
    np.vstack((X.flatten(), Y.flatten())).T, mean=mu_ex.flatten(), cov=cov_ex
).reshape(X.shape)

fig, ax = plt.subplots()
ax.scatter(Xh_0[0], Xh_0[1], s=0.3, c="w")
ax.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin="lower")
plt.show()
