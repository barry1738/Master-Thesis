"""
1D forward Ornstein-Uhenbeck process
dXt  = -beta*Xt*dt + sigma*dWt
X(0) = X0 (const.)
------------------------------
mean = X0*exp(-beta*t)
var  = sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# numerical setup
T  = 2    # terminal time
M  = 100  # number of iterations
dt = T / M  # time step size
N  = 2000 # number of particles
# ... parameters in f and g ...
beta = 1
sigma = 0.5
# ... initial condition ...
X_0 = 5
# ... exact mean and std at T ...
mu_ex = X_0 * np.exp(-beta * T)
std_ex = np.sqrt(sigma**2 / (2 * beta) * (1 - np.exp(-2 * beta * T)))


# SDE setup
def f(x, t):
    return -beta * x

def g(x, t):
    return sigma

# Euler-Maruyama method
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, 0] = X_0

for i in range(M):
    ti = i*dt
    Xh_0[:, i+1] = Xh_0[:, i] \
                 + f(Xh_0[:, i], ti) * dt \
                 + g(Xh_0[:, i], ti) * np.sqrt(dt) * np.random.randn(N)

# Compute mean and std from discrete data
mu_sde = np.mean(Xh_0[:,-1])
std_sde = np.sqrt(np.sum((Xh_0[:, -1] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact mean: {mu_ex:.6f}")
print(f"Numer.mean: {mu_sde:.6f}")
print("--------------------")
print(f"Exact std : {std_ex:.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].set_ylim([-10, 10])
axs[0].set_xlabel('time')
axs[0].set_ylabel('X(t)')

axs[1].plot(-0.1*np.ones_like(Xh_0[:, -1]), Xh_0[:, -1], 'k.')
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_ex, std_ex), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()