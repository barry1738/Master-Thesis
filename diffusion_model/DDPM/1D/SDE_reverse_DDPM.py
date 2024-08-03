"""
1D reverse DDPM process
X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
X(0) = X0 (const.)
------------------------------
mean = sqrt( bar{alpha}_k*X0 )
var  = 1 - bar{alpha}_k
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# numerical setup
T = 10  # terminal time
M = 11  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = np.sort(np.random.rand(M))
alpha = 1 - beta
alpha_bar = np.zeros(M)
alpha_bar[0] = alpha[0]
for i in range(1, M):
    alpha_bar[i] = alpha[i] * alpha_bar[i-1]
print(f'beta = {beta[:10]}')
print(f'alpha = {alpha[:10]}')
print(f'alpha_bar = {alpha_bar[:10]}')
# ... check approximations ...
sigma_tilde = np.append(
    beta[0], np.sqrt((1 - alpha[1:]) * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]))
)
# ... exact mean and std ...
X_0 = 2
# ... initial condition ...
mu_ex = np.sqrt(alpha_bar[-1]) * X_0
std_ex = np.sqrt(1 - alpha_bar[-1])

# SDE setup
def mu(t):
    return np.exp(-t ** 2 / 4) * X_0
def std(t):
    return np.sqrt(1 - np.exp(-t ** 2 / 2))
def s(x, t):
    return -(x - mu(t)) / std(t) ** 2
def eps(x, t):
    return -std(t) * s(x, t)

# Euler-Maruyama method (backward)
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, -1] = np.random.normal(mu_ex, std_ex, N)

for i in range(M, 0, -1):
    print(f'Iter: {i}')
    ti = i * dt
    Xh_0[:, i - 1] = 1 / np.sqrt(alpha[i]) * (
        Xh_0[:, i]
        - (1 - alpha[i]) / np.sqrt(1 - alpha_bar[i]) * eps(Xh_0[:, i], ti)
    ) + sigma_tilde[i] * np.random.randn(N)

print('end')
# Compute mean and std from discrete data
mu_sde = np.mean(Xh_0[:, 0])
std_sde = np.sqrt(np.sum((Xh_0[:, 0] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact mean: {X_0:.6f}")
print(f"Numer.mean: {mu_sde:.6f}")
print("--------------------")
print(f"Exact std : {0:.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].invert_xaxis()
axs[0].set_ylim([-10, 10])
axs[0].set_title('Reverse')
axs[0].set_xlabel('time')
axs[0].set_ylabel(r'$\bar{X}(t)$')
axs[1].plot(-0.1*np.ones_like(Xh_0[:, -1]), Xh_0[:, -1], 'k.')
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_ex, std_ex), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
# plt.show()