"""
Taylor-Green Vortex flow using Physics-Informed Neural Networks (PINNs)

Incompressible Navier-Stokes Equations:
u_tt + u*u_x - v*u_y = - p_x - RE*Δu
v_tt + u*v_x + v*v_y = - p_y - RE*Δv
u_x + v_y = 0

Projection Method:
Step 1: Predict the intermediate velocity field (u*, v*) using the neural network
Step 2: Project the intermediate velocity field onto the space of divergence-free fields
Step 3: Update the velocity and pressure fields
"""


import torch
import scipy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad
from mesh_generator import CreateMesh
from utilities_2 import exact_sol


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define the neural network
class PinnModel(nn.Module):
    def __init__(self, layers):
        super(PinnModel, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x, y):
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    

def prediction_step(model, x, y, prev_value):
    """The prediction step of the projection method"""

    

# def update_previous_value(prev_value, u, v, p):

    

def main():
    # Define the neural network
    model = PinnModel([2, 20, 20, 1]).to(device)
    params = dict(model.named_parameters())
    print(model)

    total_params = model.num_total_params()
    print(f"Total number of parameters: {total_params}")

    # Define the training data
    x_inner, y_inner = CreateMesh().inner_points(100)
    x_bd, y_bd = CreateMesh().boundary_points(100)
    nx, ny = CreateMesh().normal_vector(x_bd, y_bd)
    u = functional_call(model, params, (x_inner, y_inner))
    print(f"Predicted u: {u.size()}")

    # Initialize the previous value
    prev_value = dict()
    prev_value["u0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "u")
    prev_value["v0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "v")
    prev_value["p0"] = exact_sol(x_inner, y_inner, 0.0 * Dt, Re, "p")
    prev_value["u1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "u")
    prev_value["v1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "v")
    prev_value["p1"] = exact_sol(x_inner, y_inner, 1.0 * Dt, Re, "p")
    prev_value["du0dx"], prev_value["du0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "u")
    prev_value["dv0dx"], prev_value["dv0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "v")
    prev_value["du1dx"], prev_value["du1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "u")
    prev_value["dv1dx"], prev_value["dv1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "v")
    prev_value["dp0dx"], prev_value["dp0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 0.0 * Dt, Re, "p")
    prev_value["dp1dx"], prev_value["dp1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner.reshape(-1), y_inner.reshape(-1), 1.0 * Dt, Re, "p")
    # reshape the values to (n, 1)
    for key in iter(prev_value):
        prev_value[key] = prev_value[key].reshape(-1, 1)
        print(f'prev_value["{key}"]: {prev_value[key].size()}')
    
    # Predict the intermediate velocity field (u*, v*)
    prediction_step(model, x, y, prev_value)

    


if __name__ == "__main__":
    Re = 400
    Dt = 0.01
    main()