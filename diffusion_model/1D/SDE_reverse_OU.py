"""
1D reverse Ornstein-Uhenbeck process
dXt  = -beta*Xt*dt + sigma*dWt
X(0) = X0 ~ N(mu_0, sigma_0^2)
------------------------------
mean = mu_0*exp(-beta*t)
var  = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
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
# ... exact mean and std at t = 0 ...
mu_0    = 2
sigma_0 = 2
# ... initial condition ...
mu  = lambda t: mu_0 * np.exp(-beta * t)  # noqa: E731
std = lambda t: np.sqrt(  # noqa: E731
    np.exp(-2 * beta * t) * sigma_0**2
    + sigma**2 / (2 * beta) * (1 - np.exp(-2 * beta * t))
)
X_0 = np.random.normal(mu(T), std(T), N)

# Ornstein-Uhenbeck process
def f(x, t):
    return -beta * x

def g(x, t):
    return sigma

def s(x, t):
    return -(x - mu(t)) / std(t) ** 2

# Euler-Maruyama method (backward)
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, -1] = X_0

for i in range(M, 0, -1):
    ti = i * dt
    Xh_0[:, i-1] = Xh_0[:, i] \
                 + (f(Xh_0[:, i],ti) - g(Xh_0[:, i],ti)**2 * s(Xh_0[:, i],ti)) * (-dt) \
                 + g(Xh_0[:, i],ti) * np.sqrt(dt) * np.random.randn(N)

# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, 0]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, 0] - mu_sde) ** 2) / N)

print("--------------------")
print(f"Exact mean: {mu_0:.6f}")
print(f"Numer.mean: {mu_sde:.6f}")
print("--------------------")
print(f"Exact std : {sigma_0:.6f}")
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
axs[1].plot(-0.1*np.ones_like(Xh_0[:, 0]), Xh_0[:, 0], 'k.')
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_0, sigma_0), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()