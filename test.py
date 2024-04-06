import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vjp

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device = ", device)


class PinnModel(nn.Module):
    def __init__(self, layers):
        super(PinnModel, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x, y):
        """Forward pass of the neural network."""
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output


def forward_dx(model, params, x, y):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (primal, y)), x
    )
    return vjpfunc(output)[0]

def forward_dy(model, params, x, y):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(
        lambda primal: functional_call(model, params, (x, primal)), y
    )
    return vjpfunc(torch.ones_like(output))[0]

def forward_dxx(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(
        lambda primal: forward_dx(model, params, primal, y), x
    )
    return vjpfunc(torch.ones_like(output))[0]

def forward_dyy(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(
        lambda primal: forward_dy(model, params, x, primal), y
    )
    return vjpfunc(torch.ones_like(output))[0]


u_star_model = PinnModel([2, 1])
v_star_model = PinnModel([2, 1])
phi_model = PinnModel([2, 1])
psi_model = PinnModel([2, 1])

u_star_model_dir = "C:\\Users\\barry\\Desktop\\TaylorGreenVortex_Streamfunction_1000_0.025\\u_star_model"
v_star_model_dir = "C:\\Users\\barry\\Desktop\\TaylorGreenVortex_Streamfunction_1000_0.025\\v_star_model"
phi_model_dir = "C:\\Users\\barry\\Desktop\\TaylorGreenVortex_Streamfunction_1000_0.025\\phi_model"
psi_model_dir = "C:\\Users\\barry\\Desktop\\TaylorGreenVortex_Streamfunction_1000_0.025\\psi_model"
data_dir = "C:\\Users\\barry\\Desktop\\TaylorGreenVortex_Streamfunction_1000_0.025\\data"

u_star_model = torch.load(u_star_model_dir + "\\u_star_40.pt", map_location=device)
v_star_model = torch.load(v_star_model_dir + "\\v_star_40.pt", map_location=device)
phi_model = torch.load(phi_model_dir + "\\phi_40.pt", map_location=device)
psi_model = torch.load(psi_model_dir + "\\psi_40.pt", map_location=device)
data = torch.load(data_dir + "\\initial_value_40.pt", map_location=device)
data_valid = torch.load(data_dir + "\\initial_value_valid_40.pt", map_location=device)
u_star_model.eval()
v_star_model.eval()
phi_model.eval()
psi_model.eval()

print(u_star_model)

# data_x, data_y = data["x_data"], data["y_data"]
data_x, data_y = data_valid["x_data"], data_valid["y_data"]
# data_u, data_v, data_p = data["u1"], data["v1"], data["p1"]
data_u, data_v, data_p = data_valid["u1"], data_valid["v1"], data_valid["p1"]

fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
ax[0].scatter(data_x, data_y, data_u, c=data_u, cmap="coolwarm", s=1)
ax[1].scatter(data_x, data_y, data_v, c=data_v, cmap="coolwarm", s=1)
ax[2].scatter(data_x, data_y, data_p, c=data_p, cmap="coolwarm", s=1)
ax[0].set_title("u")
ax[1].set_title("v")
ax[2].set_title("p")
plt.show()


u_star = u_star_model(torch.tensor(data_x), torch.tensor(data_y)).detach().numpy()
v_star = v_star_model(torch.tensor(data_x), torch.tensor(data_y)).detach().numpy()
phi = phi_model(torch.tensor(data_x), torch.tensor(data_y)).detach().numpy()
psi = psi_model(torch.tensor(data_x), torch.tensor(data_y)).detach().numpy()

fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
ax[0][0].scatter(data_x, data_y, u_star, c=u_star, cmap="coolwarm", s=1)
ax[0][1].scatter(data_x, data_y, v_star, c=v_star, cmap="coolwarm", s=1)
ax[1][0].scatter(data_x, data_y, phi, c=phi, cmap="coolwarm", s=1)
ax[1][1].scatter(data_x, data_y, psi, c=psi, cmap="coolwarm", s=1)
ax[0][0].set_title("u_star")
ax[0][1].set_title("v_star")
ax[1][0].set_title("phi")
ax[1][1].set_title("psi")
plt.show()