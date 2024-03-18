import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
from torch.func import functional_call, vjp, grad, vmap


torch.set_default_dtype(torch.float64)


class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        # input layer
        self.ln_in = nn.Linear(in_dim, h_dim[0])

        self.hidden = nn.ModuleList()
        # hidden layers
        for i in range(len(h_dim) - 1):
            self.hidden.append(nn.Linear(h_dim[i], h_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(h_dim[-1], out_dim, bias=False)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, x, y, z):
        input = torch.hstack((x, y, z))
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

class CreateMesh:
    def __init__(self, interface_func, *, radius=1):
        self.func = interface_func
        self.r = radius

    def domain_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution within a circle"""
        radius = torch.tensor(self.r * np.sqrt(qmc.LatinHypercube(d=1).random(n=n)))
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
        x = xc + radius * torch.cos(theta)
        y = yc + radius * torch.sin(theta)
        return x, y

    def boundary_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a circle"""
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
        x = xc + self.r * torch.cos(theta)
        y = yc + self.r * torch.sin(theta)
        return x, y

    def interface_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a polar curve"""
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=n))
        radius = self.func(theta)
        x = xc + radius * torch.cos(theta)
        y = yc + radius * torch.sin(theta)
        return x, y

    def sign(self, x, y):
        """Check the points inside the polar curve, return z = -1 if inside, z = 1 if outside"""
        dist = torch.sqrt(x**2 + y**2) - self.func(torch.atan2(y, x))
        z = torch.where(dist > 0, 1, -1)
        return z

    def compute_boundary_normal_vec(self, x, y):
        """Compute the boundary normal vector"""
        nx = 2 * x / 1
        ny = 2 * y / 1
        dist = torch.sqrt(nx**2 + ny**2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny

    def compute_interface_normal_vec(self, x, y):
        """Compute the interface normal vector"""
        theta = torch.atan2(y, x)
        r = self.func(theta)
        drdt = vmap(grad(self.func))(theta.reshape(-1)).view(-1, 1)
        nx = drdt * torch.sin(theta) + r * torch.cos(theta)
        ny = -drdt * torch.cos(theta) + r * torch.sin(theta)
        dist = torch.sqrt(nx**2 + ny**2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny

    def compute_interface_curvature(self, x, y):
        """Compute the interface curvature"""
        theta = torch.atan2(y, x)
        r = self.func(theta)
        drdt = vmap(grad(self.func))(theta.reshape(-1)).view(-1, 1)
        d2rdt2 = vmap(grad(grad(self.func)))(theta.reshape(-1)).view(-1, 1)
        dxdt = drdt * torch.cos(theta) - r * torch.sin(theta)
        dydt = drdt * torch.sin(theta) + r * torch.cos(theta)
        d2xdt2 = (
            d2rdt2 * torch.cos(theta)
            - 2 * drdt * torch.sin(theta)
            - r * torch.cos(theta)
        )
        d2ydt2 = (
            d2rdt2 * torch.sin(theta)
            + 2 * drdt * torch.cos(theta)
            - r * torch.sin(theta)
        )
        curvature = (dxdt * d2ydt2 - dydt * d2xdt2) / (dxdt**2 + dydt**2) ** (3 / 2)
        return curvature


def forward_dx(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (primal, y, z)), x
    )
    return vjpfunc(torch.ones_like(output))[0]


def forward_dy(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (x, primal, z)), y
    )
    return vjpfunc(torch.ones_like(output))[0]


def forward_dxx(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: forward_dx(model, params, primal, y, z), x)
    return vjpfunc(torch.ones_like(output))[0]


def forward_dyy(model, params, x, y, z):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: forward_dy(model, params, x, primal, z), y)
    return vjpfunc(torch.ones_like(output))[0]


def predict(model, params, x, y, z):
    """Compute the model output."""
    return functional_call(model, params, (x, y, z)) / torch.sqrt(x**2 + y**2)


def predict_dx(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: predict(model, params, primal, y, z), x)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dy(model, params, x, y, z):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: predict(model, params, x, primal, z), y)
    return vjpfunc(torch.ones_like(output))[0]


# Load the model
model = torch.load(
    "C:\\Users\\barry\\Desktop\\2024_03_18\\cos_2t\\model_cos_6t.pt",
    map_location=torch.device("cpu"),
)
model.eval()
print(model)

params = dict(model.named_parameters())


# Create the mesh
mesh = CreateMesh(
    interface_func=lambda theta: 1 + torch.cos(2 * theta) / 10, radius=1.5
)
plot_x, plot_y = mesh.domain_points(50000)
plot_z = mesh.sign(plot_x, plot_y)
# plot_z = torch.ones_like(plot_x)
# result = functional_call(model, params, (plot_x, plot_y, plot_z)).detach().numpy()
result = predict(model, params, plot_x, plot_y, plot_z).detach().numpy()

# Plot the resutls on the interface
theta = torch.tensor(np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1))
r = mesh.func(theta)
if_x = r * torch.cos(theta)
if_y = r * torch.sin(theta)
nx, ny = mesh.compute_interface_normal_vec(if_x, if_y)
k_if = mesh.compute_interface_curvature(if_x, if_y)

z_outer = 1.0 * torch.ones_like(if_x)
# dfdx_outer = forward_dx(model, params, if_x, if_y, z_outer)
# dfdy_outer = forward_dy(model, params, if_x, if_y, z_outer)
dfdx_outer = predict_dx(model, params, if_x, if_y, z_outer)
dfdy_outer = predict_dy(model, params, if_x, if_y, z_outer)
pred = nx * dfdx_outer + ny * dfdy_outer


# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
sc = ax1.scatter(plot_x, plot_y, result, c=result, cmap="coolwarm", s=1)
ax1.axes.zaxis.set_ticklabels([])
fig.colorbar(sc, shrink=0.5, aspect=7, pad=0.02)
ax1.set_title(r"$1+0.1\cos(2\theta)$, z=1")
# plt.savefig("C:\\Users\\barry\\Desktop\\cos_2t.png", dpi=300)
# plt.show()

ax2 = fig.add_subplot(1, 2, 2)
sc2 = ax2.plot(theta, pred.detach().numpy())
# ax2.axis("equal")
plt.show()

# Save the results to csv
# df = pd.DataFrame(
#     {
#         "theta": theta.detach().numpy().flatten(),
#         "pred": pred.detach().numpy().flatten(),
#         "k_if": k_if.detach().numpy().flatten(),
#     }
# )
# df.to_csv(
#     "C:\\Users\\barry\\Desktop\\cos6t_data.csv",
#     index=False,
#     header=False,
#     encoding='utf-8'
# )