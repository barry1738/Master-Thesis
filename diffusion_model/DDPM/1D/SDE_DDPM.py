"""
1D forward DDPM process
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
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = np.arange(1, M + 1) * dt ** 2
alpha     = 1 - beta
alpha_bar = np.array([np.prod(alpha[:i+1]) for i in range(M)])
# ... initial condition ...
X_0 = 5
# ... exact mean and std ...
mu_ex = np.sqrt(alpha_bar[-1]) * X_0
std_ex = np.sqrt(1 - alpha_bar[-1])

# Euler-Maruyama method
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, 0] = X_0

for i in range(M):
    ti = i * dt
    Xh_0[:, i + 1] = np.sqrt(1 - beta[i]) * Xh_0[:, i] + np.sqrt(beta[i]) * np.random.randn(N)

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, -1]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, -1] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact.mean: {mu_ex:.6f}")
print(f'Numer.mean: {mu_sde:.6f}')
print("--------------------")
print(f"Exact.std : {std_ex:.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].set_ylim([-10, 10])
axs[0].set_xlabel("time")
axs[0].set_ylabel("X(t)")
axs[0].set_title("Reverse")

axs[1].plot(-0.1 * np.ones_like(Xh_0[:, -1]), Xh_0[:, -1], "k.")
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_ex, std_ex), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()

