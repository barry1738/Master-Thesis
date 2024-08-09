"""
MNIST reverse DDPM process (SAMPLING)
X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
X(0) = X0 (const.)
------------------------------
mean = sqrt( bar{alpha}_k*X0 )
var  = 1 - bar{alpha}_k
------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# numerical setup
d = 784  # problem dimension
T = 10  # terminal time
M = 1000  # number of iterations
dt = T / M  # time step
N = 1  # number of particles
# ... parameters in DDPM ...
beta = np.arange(1.0, M + 1, 1.0) * dt**2
alpha = 1 - beta
alpha_bar = np.zeros(M)
alpha_bar[0] = alpha[0]
for i in range(1, M):
    alpha_bar[i] = alpha_bar[i - 1] * alpha[i]
sigma_tilde = np.append(
    beta[0], np.sqrt((1 - alpha[1:]) * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]))
)
# ... initial condition ...
X_0 = np.loadtxt("diffusion_model/DDPM/MNIST/X_0.csv").reshape(-1, 1)
mu_ex = np.zeros((d, 1))
std_ex = 1
Xh_0 = np.zeros((d, M))
Xh_0[:, -1] = np.random.multivariate_normal(mu_ex.flatten(), std_ex**2 * np.eye(d), N).flatten()

# SDE setup
mu = lambda t: np.exp(-(t ** 2) / 4) * X_0
std = lambda t: np.sqrt(1 - np.exp(-t ** 2 / 2))
s = lambda x, t: -(x - mu(t)) / std(t) ** 2
eps = lambda x, t: -std(t) * s(x, t)


def init():
    ax.axis("off")


def update(frame):
    ax.clear()
    # Euler-Maruyama method (backward)
    ti = frame * dt
    Xh_0[:, frame - 1] = (
        1
        / np.sqrt(alpha[frame])
        * (
            Xh_0[:, frame].reshape(-1, 1)
            - (1 - alpha[frame])
            / np.sqrt(1 - alpha_bar[frame])
            * eps(Xh_0[:, frame].reshape(-1, 1), ti)
        )
        + sigma_tilde[frame] * np.random.randn(d, N)
    ).flatten()

    ax.imshow(Xh_0[:, frame].reshape(28, 28).T, cmap="gray")
    ax.set_title(f"t = {frame}")
    ax.axis("off")


fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig, update, frames=range(M - 1, 0, -1), interval=1, init_func=init, repeat=False
)
ani.save("C:\\Users\\barry\\Desktop\\MNIST.gif", writer="pillow", fps=100)

# Compute mean and std from discrete data
mu_sde = Xh_0[:, 0]
cov_sde = np.cov(Xh_0[:, 0].T) * (1 - 1 / N)

norm = np.linalg.norm(mu_sde - X_0.flatten(), ord=np.inf)
print(f"norm of difference in mean = {norm:.6f}")
