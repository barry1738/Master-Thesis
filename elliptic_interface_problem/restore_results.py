import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
from torch.func import functional_call, vjp


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
    def __init__(self) -> None:
        pass

    def domain_points(self, n, *, xc=0, yc=0, r=2):
        """Uniform random distribution within a circle"""
        radius = r * np.sqrt(qmc.LatinHypercube(d=1).random(n=n))
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)

    def boundary_points(self, n, *, xc=0, yc=0, r=2):
        """Uniform random distribution on a circle"""
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        x = xc + r * np.cos(theta)
        y = yc + r * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)

    def interface_points(self, n, *, xc=0, yc=0):
        """Uniform random distribution on a polar curve"""
        theta = 2 * np.pi * qmc.LatinHypercube(d=1).random(n=n)
        # radius = 1 + np.cos(2 * theta) / 10
        radius = 1
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)
        return torch.tensor(x), torch.tensor(y)

    def sign(self, x, y):
        """Check the points inside the polar curve, return z = -1 if inside, z = 1 if outside"""
        # dist = torch.sqrt(x**2 + y**2) - (1 + torch.cos(2 * torch.atan2(y, x)) / 10)
        dist = torch.sqrt(x ** 2 + y ** 2) - 1
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
        # ri = 1 + torch.cos(2 * theta) / 10
        # dri = -2 * torch.sin(2 * theta) / 10
        ri = 1
        dri = 0
        nx = dri * torch.sin(theta) + ri * torch.cos(theta)
        ny = -dri * torch.cos(theta) + ri * torch.sin(theta)
        dist = torch.sqrt(nx**2 + ny**2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny

    def compute_interface_curvature(self, x, y):
        """Compute the interface curvature"""
        theta = torch.atan2(y, x)
        # r = 1 + torch.cos(2 * theta) / 10
        # drdt = -2 * torch.sin(2 * theta) / 10
        # d2rdt2 = -4 * torch.cos(2 * theta) / 10
        r = 1
        drdt = 0
        d2rdt2 = 0
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


# Load the model
model = Model(3, [50, 50], 1)
# model = torch.load('model6.pt', map_location=torch.device('cpu'))
model.load_state_dict(torch.load("model6.pt"))
model.eval()
print(model)

params = dict(model.named_parameters())



# Create the mesh
mesh = CreateMesh()
plot_x, plot_y = mesh.domain_points(50000)
plot_z = mesh.sign(plot_x, plot_y)
result = functional_call(model, params, (plot_x, plot_y, plot_z)).detach().numpy()

# Plot the resutls on the interface
theta = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
r = 1 / 2 + np.cos(3 * theta) / 10
if_x = torch.tensor(r * np.cos(theta))
if_y = torch.tensor(r * np.sin(theta))
nx, ny = mesh.compute_interface_normal_vec(if_x, if_y)

z_outer = 1.0 * torch.ones_like(if_x)
dfdx_outer = forward_dx(model, params, if_x, if_y, z_outer)
dfdy_outer = forward_dy(model, params, if_x, if_y, z_outer)
pred = nx * dfdx_outer + ny * dfdy_outer


# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
sc = ax1.scatter(plot_x, plot_y, result, c=result, cmap="coolwarm", s=1)
ax1.axes.zaxis.set_ticklabels([])
fig.colorbar(sc, shrink=0.5, aspect=7, pad=0.02)
# plt.show()

ax2 = fig.add_subplot(1, 2, 2)
sc2 = ax2.plot(theta, pred.detach().numpy())
# ax2.axis("equal")
plt.show()