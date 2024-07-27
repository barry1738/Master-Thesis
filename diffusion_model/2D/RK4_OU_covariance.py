import numpy as np

def RK4_OU(endT, F, G, S0):
    # Numerical Parameters
    dt = 0.01
    nt = np.ceil(endT/dt)
    H = lambda t, S: F @ S + S @ F.T + G @ G.T

    # Initial condition
    S = S0

    # Main loop
    for i in range(int(nt)):
        k1 = dt * H((i - 1) * dt, S)
        k2 = dt * H((i - 0.5) * dt, S + k1 / 2)
        k3 = dt * H((i - 0.5) * dt, S + k2 / 2)
        k4 = dt * H(i * dt, S + k3)
        S = S + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return S