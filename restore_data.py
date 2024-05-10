import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vjp

torch.set_default_dtype(torch.float64)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# print("device = ", device)


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


def exact_solution(x, y, t, type):
    """calculate analytical solution
    u(x,y,t) = -cos(πx)sin(πy)exp(-2π²t/RE)
    v(x,y,t) =  sin(πx)cos(πy)exp(-2π²t/RE)
    p(x,y,t) = -0.25(cos(2πx)+cos(2πy))exp(-4π²t/RE)
    """
    match type:
        case "u":
            return (
                -torch.cos(torch.pi * x)
                * torch.sin(torch.pi * y)
                * torch.exp(torch.tensor(-2 * torch.pi**2 * t / Re))
            )
        case "v":
            return (
                torch.sin(torch.pi * x)
                * torch.cos(torch.pi * y)
                * torch.exp(torch.tensor(-2 * torch.pi**2 * t / Re))
            )
        case "p":
            return (
                -0.25
                * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))
                * torch.exp(torch.tensor(-4 * torch.pi**2 * t / Re))
            )


def main():
    # Initialize the model
    u_star_model = PinnModel([2, 1])
    v_star_model = PinnModel([2, 1])
    phi_model = PinnModel([2, 1])
    psi_model = PinnModel([2, 1])

    pwd = "C:\\Users\\barry\\OneDrive\\thesis\\Data\\TaylorGreenVortex_Streamfunction_square_5s\\"
    models_dir = pwd + "models\\"
    params_dir = pwd + "params\\"

    # Load the models
    u_star_model = torch.load(models_dir + "u_star_model.pt", map_location=torch.device("cpu"))
    v_star_model = torch.load(models_dir + "v_star_model.pt", map_location=torch.device("cpu"))
    phi_model = torch.load(models_dir + "phi_model.pt", map_location=torch.device("cpu"))
    psi_model = torch.load(models_dir + "psi_model.pt", map_location=torch.device("cpu"))
    u_model = torch.load(models_dir + "u_model.pt", map_location=torch.device("cpu"))
    v_model = torch.load(models_dir + "v_model.pt", map_location=torch.device("cpu"))
    p_model = torch.load(models_dir + "p_model.pt", map_location=torch.device("cpu"))

    # Load the parameters
    u_star_params = torch.load(params_dir + f"u_star_params\\u_star_{step}.pt", map_location=torch.device("cpu"))
    v_star_params = torch.load(params_dir + f"v_star_params\\v_star_{step}.pt", map_location=torch.device("cpu"))
    phi_params = torch.load(params_dir + f"phi_params\\phi_{step}.pt", map_location=torch.device("cpu"))
    psi_params = torch.load(params_dir + f"psi_params\\psi_{step}.pt", map_location=torch.device("cpu"))
    u_params = torch.load(params_dir + f"u_params\\u_{step}.pt", map_location=torch.device("cpu"))
    v_params = torch.load(params_dir + f"v_params\\v_{step}.pt", map_location=torch.device("cpu"))
    p_params = torch.load(params_dir + f"p_params\\p_{step}.pt", map_location=torch.device("cpu"))

    # Pin p parameters
    p_params["linear_layers.2.bias"] += (
        exact_solution(torch.tensor([[0.0]]), torch.tensor([[0.0]]), step * time_step, "p") - 
        functional_call(p_model, p_params, (torch.tensor([[0.0]]), torch.tensor([[0.0]])))
    ).reshape(-1)
    
    u_star_model.load_state_dict(u_star_params)
    v_star_model.load_state_dict(v_star_params)
    phi_model.load_state_dict(phi_params)
    psi_model.load_state_dict(psi_params)
    u_model.load_state_dict(u_params)
    v_model.load_state_dict(v_params)
    p_model.load_state_dict(p_params)

    # Set the models to evaluation mode
    u_star_model.eval()
    v_star_model.eval()
    phi_model.eval()
    psi_model.eval()
    u_model.eval()
    v_model.eval()
    p_model.eval()

    print(u_star_model)

    # Restore the data
    x_plot, y_plot = torch.meshgrid(
        torch.linspace(0, 1, 200), torch.linspace(0, 1, 200), indexing="xy"
    )
    x_plot, y_plot = x_plot.reshape(-1, 1), y_plot.reshape(-1, 1)

    phi_pred = functional_call(phi_model, phi_params, (x_plot, y_plot)).detach().numpy()
    psi_pred = functional_call(psi_model, psi_params, (x_plot, y_plot)).detach().numpy()
    u_pred = functional_call(u_model, u_params, (x_plot, y_plot)).detach().numpy()
    v_pred = functional_call(v_model, v_params, (x_plot, y_plot)).detach().numpy()
    p_pred = functional_call(p_model, p_params, (x_plot, y_plot)).detach().numpy()
    u_exact = exact_solution(x_plot, y_plot, step * time_step, "u").detach().numpy()
    v_exact = exact_solution(x_plot, y_plot, step * time_step, "v").detach().numpy()
    p_exact = exact_solution(x_plot, y_plot, step * time_step, "p").detach().numpy()
    u_error = np.abs(u_pred - u_exact)
    v_error = np.abs(v_pred - v_exact)
    p_error = np.abs(p_pred - p_exact)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x_plot, y_plot, p_error, c=p_error, cmap="coolwarm", s=1)
    ax.set_title("p error")
    plt.show()


if __name__ == "__main__":
    Re = 1000
    step = 10
    time_step = 0.02

    main()