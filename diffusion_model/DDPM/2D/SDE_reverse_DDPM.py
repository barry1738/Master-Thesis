"""
2D reverse DDPM process
X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
X(0) = X0 (const.)
------------------------------
mean = sqrt( bar{alpha}_k*X0 )
var  = 1 - bar{alpha}_k
------------------------------
"""

import numpy as np

# FIXME: alpha[-1] and alpha_bar[-1] is zero, which causes division by zero


# numerical setup
T = 3  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = np.arange(1.0, M + 1, 1.0) * dt**2
alpha = 1 - beta
alpha_bar = np.zeros(M)
alpha_bar[0] = alpha[0]
for i in range(1, M):
    alpha_bar[i] = alpha_bar[i - 1] * alpha[i]
# ... check approximation ...
sigma_tilde = np.append(
    beta[0], np.sqrt((1 - alpha_bar[1:]) * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]))
)
# ... exact mean and std ...
X_0 = np.array([[1], [2]])
# ... initial condition ...
mu_ex = np.sqrt(alpha_bar[-1]) * X_0
std_ex = np.sqrt(1 - alpha_bar[-1])
Xh_0 = np.random.multivariate_normal(mu_ex.reshape(-1), std_ex**2 * np.eye(2), N).T
# SDE setup
mu = lambda t: np.exp(-(t**2) / 4) * X_0
std = lambda t: np.sqrt(1 - np.exp(-t**2 / 2))
s = lambda x, t: -(x - mu(t)) / std(t)**2
eps = lambda x, t: -std(t) * s(x, t)

# Euler-Maruyama method
for i in range(M - 1, 0, -1):
    ti = i * dt
    Xh_0 = 1 / np.sqrt(alpha[i]) * (
        Xh_0 - (1 - alpha[i]) / np.sqrt(1 - alpha_bar[i]) * eps(Xh_0, ti)
    ) + sigma_tilde[i] * np.random.randn(2, N)

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