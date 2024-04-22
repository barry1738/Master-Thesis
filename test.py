import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vjp

x = np.linspace(0, 1.5, 1000).reshape(-1, 1)
func1 = np.sin(2 * np.pi * x)
func2 = np.cos(2 * np.pi * x)
func3 = np.exp(-10 * (x - 1.5)**2)

fig, ax = plt.subplots()
ax.plot(x, func1, label="sin(2 * pi * x)")
ax.plot(x, func2, label="cos(2 * pi * x)")
ax.plot(x, func3, label="exp(-10 * (x - 0.5)**2)")
ax.legend()
plt.show()