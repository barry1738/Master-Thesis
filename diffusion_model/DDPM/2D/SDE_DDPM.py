"""
2D forward DDPM process
X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
X(0) = X0 (const.)
------------------------------ 
mean = sqrt( bar{alpha}_k*X0 )
var  = 1 - bar{alpha}_k
------------------------------
"""

import numpy as np


# numerical setup
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles
# ... parameters in f and g ...
beta = np.arange(1.0, M + 1, 1.0) * dt**2
alpha = 1 - beta
alpha_bar = np.zeros(M)
alpha_bar[0] = alpha[0]
for i in range(1, M):
    alpha_bar[i] = alpha_bar[i - 1] * alpha[i]
# ... initial condition ...
X0 = np.array([[5], [6]])
Xh_0 = np.zeros((2, N)) + X0
# ... exact mean and std at t = T ...
mu_ex = np.sqrt(alpha_bar[-1]) * X0
cov_ex = (1 - alpha_bar[-1]) * np.eye(2)

# Euler-Maruyama method
for i in range(M):
    ti = i * dt
    Xh_0 = np.sqrt(1 - beta[i]) * Xh_0 + np.sqrt(beta[i]) * np.random.randn(2, N)

#  Compute mean and std from discrete data
mu_sde = np.sum(Xh_0, axis=1) / N
cov_sde = np.cov(Xh_0) * (1 - 1 / N)

# Output
print('------------------------')
print(f"Exact.mean = \n{mu_ex.reshape(-1)}")
print(f"Numer.mean = \n{mu_sde}")
print('------------------------')
print(f"Exact.cov = \n{cov_ex}")
print(f"Numer.cov = \n{cov_sde}")
print('------------------------')
