import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# FIXME: alpha[-1] and alpha_bar[-1] is zero, which causes division by zero


# numerical setup
T  = 10   # terminal time
M  = 1000 # number of iterations
dt = T / M  # time step size
N  = 2000 # number of particles
# ... parameters in DDPM ...
beta = np.arange(1, M + 1) * dt ** 2
alpha     = 1 - beta
alpha_bar = np.zeros(M)
alpha_bar[0] = alpha[0]
for i in range(1, M):
    alpha_bar[i] = alpha[i] * alpha_bar[i - 1]
sigma_bar = np.append(
    beta[0], np.sqrt((1 - alpha[1:]) * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]))
)
# print(f"beta: {beta.shape}")
# print(f"alpha: {alpha.shape}")
# print(f"alpha_bar: {alpha_bar.shape}")
# print(f'sigma_bar: {sigma_bar.shape}')
# ... exact mean and std ...
mu_0    = 3
sigma_0 = 1.5
# ... check approximations ...
# std = np.sqrt(
#     np.exp(-0.5 * np.arange(1, M + 1) * dt**2) * sigma_0**2
#     + 1
#     - np.exp(-0.5 * np.arange(1, M + 1) * dt**2)
# )

# ... score and noise function ...
mu = lambda t: np.exp(-(t**2) / 4) * mu_0
std = lambda t: np.sqrt(np.exp(-(t**2) / 2) * sigma_0**2 + 1 - np.exp(-(t**2) / 2))
s = lambda x, t: -(x - mu(t)) / std(t) ** 2
sigma_tilde = lambda t: np.sqrt(1 - np.exp(-(t**2) / 2))
eps = lambda x, t: -sigma_tilde(t) * s(x, t)


# ... initial condition ...
mu_M = np.sqrt(alpha_bar[-1]) * mu_0
std_M = np.sqrt(alpha_bar[-1] * sigma_0 ** 2 + 1 - alpha_bar[-1])
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, -1] = np.random.normal(mu_M, std_M, N)

# ... main loop ...
for i in range(M - 1, -1, -1):
    # print(f"i = {i}")
    ti = i * dt

    # Xh_0[:, i] = 1 / np.sqrt(alpha[i]) * (
    #     Xh_0[:, i + 1] + (1 - alpha[i]) * s(Xh_0[:, i + 1], ti)
    # ) + sigma_bar[i] * np.random.randn(N)

    Xh_0[:, i] = 1 / np.sqrt(alpha[i]) * (
        Xh_0[:, i + 1]
        - (1 - alpha[i]) / np.sqrt(1 - alpha_bar[i]) * eps(Xh_0[:, i + 1], ti)
    ) + sigma_bar[i] * np.random.randn(N)

    # Xh_0[:, i] = (
    #     1 / np.sqrt(alpha[i]) * Xh_0[:, i + 1]
    #     - (1 - alpha[i]) / np.sqrt(1 - alpha_bar[i]) * eps(Xh_0[:, i + 1], ti)
    #     + sigma_bar[i] * np.random.randn(N)
    # )


# Compute mean and std from discrete data
mu_sde = np.sum(Xh_0[:, 0]) / N
std_sde = np.sqrt(np.sum((Xh_0[:, 0] - mu_sde) ** 2) / N)

print("--------------------")
print(f'Exact.mean: {mu_0:.6f}')
print(f'Numer.mean: {mu_sde:.6f}')
print("--------------------")
print(f"Exact.std : {std(0):.6f}")
print(f"Numer.std : {std_sde:.6f}")
print("--------------------")

# Output
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(np.linspace(0, T, M + 1), Xh_0.T)
axs[0].invert_xaxis()
axs[0].set_ylim([-10, 10])
axs[0].set_xlabel("time")
axs[0].set_ylabel(r"$\bar{X}$(t)")
axs[0].set_title("Reverse")

axs[1].scatter(-0.1 * np.ones_like(Xh_0[:, 0]), Xh_0[:, 0], c="k", s=1)
axs[1].plot(norm.pdf(np.arange(-10, 10, 0.1), mu_0, std(0)), np.arange(-10, 10, 0.1))
axs[1].set_axis_off()
plt.show()

