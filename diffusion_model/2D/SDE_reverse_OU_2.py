"""
2D forward stochastic process
dXt  = F*Xt*dt + G*dWt
X(0) = X0 ~ N(mu_0, sigma_0^2)
------------------------------
mean(t) = exp(t*F)*mu_0
Cov(t)  = SIGMA(t) (solved by RK4)
score(x,t) = -inv(SIGMA(t))*( x - mean(t) )
-------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import expm, solve
from RK4_OU_covariance import RK4_OU


# Numerical Setup
T = 2  # Terminal time
M = 100  # Number of iterations
dt = T / M  # Time step size
N = 2000  # Number of particles

# ... construct F and G ...
F = np.random.randn(2, 2)
F = - F @ F.T
G = np.random.randn(2, 2)
G = G @ G.T

# ... exact mean and std at t = 0 ...
mu_0 = np.array([[1], [2]])
SIGMA_0 = np.random.randn(2, 2)
SIGMA_0 = SIGMA_0.T @ SIGMA_0

# ... initial condition ...
mu = lambda t: expm(t * F) @ mu_0
SIGMA = lambda t: RK4_OU(t, F, G, SIGMA_0)
Xh_0 = np.random.multivariate_normal(mu(T).flatten(), SIGMA(T), N).T

# Ornstein-Uhenbeck process
f = lambda x, t: F @ x
g = lambda x, t: G
s = lambda x, t: solve(-SIGMA(t), x - mu(t))

# Euler-Maruyama method
for i in range(M, 0, -1):
    ti = i * dt
    Xh_0 = (
        Xh_0
        + (f(Xh_0, ti) - g(Xh_0, ti) @ g(Xh_0, ti).T @ s(Xh_0, ti)) * (-dt)
        + g(Xh_0, ti) @ np.random.randn(2, N) * np.sqrt(dt)
    )

# Compute mean and covariance from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print(f"Exact.mean = \n{mu_0.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print(f"Exact.cov = \n{SIGMA_0}")
print(f"Numer.cov = \n{cov_sde}")

xx = np.arange(mu_0[0, 0] - 5, mu_0[0, 0] + 5, 0.1)
yy = np.arange(mu_0[1, 0] - 5, mu_0[1, 0] + 5, 0.1)
X, Y = np.meshgrid(xx, yy, indexing="xy")

Z = multivariate_normal.pdf(
    np.vstack((X.flatten(), Y.flatten())).T,
    mean=mu_0.flatten(),
    cov=SIGMA_0,
).reshape(X.shape)

fig, ax = plt.subplots()
ax.scatter(Xh_0[0], Xh_0[1], s=0.3, c="w")
ax.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin="lower")
plt.show()
