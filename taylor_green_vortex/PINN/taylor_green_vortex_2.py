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
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad
from mesh_generator import CreateMesh
from utilities_2 import exact_sol, qr_decomposition, cholesky


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
        """Forward pass of the neural network."""
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output
    
    def predict(self, model, params, x, y):
        """Predict the model output."""
        return functional_call(model, params, (x, y))
    
    def predict_dx(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dy(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dxx(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: self.predict_dx(model, params, primal, y), x)
        return vjpfunc(torch.ones_like(output))[0]

    def predict_dyy(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: self.predict_dy(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    

def prediction_step(model, training_data, prev_val, prev_val_v, step):
    """The prediction step of the projection method"""
    # Unpack the training data
    x_inner, y_inner = training_data[0]
    x_bd, y_bd = training_data[1]
    x_inner_v, y_inner_v = training_data[2]
    x_bd_v, y_bd_v = training_data[3]
    nx, ny = training_data[4]

    # Define the parameters
    u_star_params = dict(model.named_parameters())
    v_star_params = dict(model.named_parameters())

    # Compute the right-hand side values
    Rf_u_inner = (
        4 * prev_val["u1"]
        - prev_val["u0"]
        - 2 * (2 * Dt) * (prev_val["u1"] * prev_val["du1dx"] + prev_val["v1"] * prev_val["du1dy"])
        + (2 * Dt) * (prev_val["u0"] * prev_val["du0dx"] + prev_val["v0"] * prev_val["du0dy"])
        - (2 * Dt) * (prev_val["dp1dx"])
    )
    Rf_v_inner = (
        4 * prev_val["v1"]
        - prev_val["v0"]
        - 2 * (2 * Dt) * (prev_val["u1"] * prev_val["dv1dx"] + prev_val["v1"] * prev_val["dv1dy"])
        + (2 * Dt) * (prev_val["u0"] * prev_val["dv0dx"] + prev_val["v0"] * prev_val["dv0dy"])
        - (2 * Dt) * (prev_val["dp1dy"])
    )
    Rf_u_inner_v = (
        4 * prev_val_v["u1"]
        - prev_val_v["u0"]
        - 2 * (2 * Dt) * (prev_val_v["u1"] * prev_val_v["du1dx"] + prev_val_v["v1"] * prev_val_v["du1dy"])
        + (2 * Dt) * (prev_val_v["u0"] * prev_val_v["du0dx"] + prev_val_v["v0"] * prev_val_v["du0dy"])
        - (2 * Dt) * (prev_val_v["dp1dx"])
    )
    Rf_v_inner_v = (
        4 * prev_val_v["v1"]
        - prev_val_v["v0"]
        - 2 * (2 * Dt) * (prev_val_v["u1"] * prev_val_v["dv1dx"] + prev_val_v["v1"] * prev_val_v["dv1dy"])
        + (2 * Dt) * (prev_val_v["u0"] * prev_val_v["dv0dx"] + prev_val_v["v0"] * prev_val_v["dv0dy"])
        - (2 * Dt) * (prev_val_v["dp1dy"])
    )
    Rf_u_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "u")
    Rf_v_bd = exact_sol(x_bd, y_bd, step * Dt, Re, "v")
    Rf_u_bd_v = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "u")
    Rf_v_bd_v = exact_sol(x_bd_v, y_bd_v, step * Dt, Re, "v")


    def compute_loss_res(model, params, x, y, Rf_inner):
        """Compute the residual loss function."""
        pred = (
            3 * model.predict(model, params, x, y) -
            (2 * Dt / Re) * (
                model.predict_dxx(model, params, x, y) +
                model.predict_dyy(model, params, x, y)
            )
        )
        loss_res = pred - Rf_inner
        return loss_res

    def compute_loss_bd(model, params, x, y, Rf_bd):
        """Compute the boundary loss function."""
        pred = model.predict(model, params, x, y)
        loss_bd = pred - Rf_bd
        return loss_bd
    
    # Start training
    Niter = 1000
    tol = 1.0e-10
    mu = 1.0e5
    savedloss = []
    saveloss_vaild = []
    alpha = 1.0
    beta = 1.0

    for iter in range(Niter):
        # Compute the jacobi matrix
        jac_res_dict = vmap(
            jacrev(compute_loss_res, argnums=(1)), in_dims=(None, 0, 0, 0), out_dims=0
        )(model, u_star_params, x_inner, y_inner, Rf_u_inner)

        jac_bd_dict = vmap(
            jacrev(compute_loss_bd, argnums=(1)), in_dims=(None, 0, 0, 0), out_dims=0
        )(model, u_star_params, x_bd, y_bd, Rf_u_bd)


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
    x_inner_v, y_inner_v = CreateMesh().inner_points(10000)
    x_bd_v, y_bd_v = CreateMesh().boundary_points(1000)
    # Compute the boundary normal vector
    nx, ny = CreateMesh().normal_vector(x_bd, y_bd)

    # Move the data to the device
    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
    x_inner_v, y_inner_v = x_inner_v.to(device), y_inner_v.to(device)
    x_bd_v, y_bd_v = x_bd_v.to(device), y_bd_v.to(device)
    nx, ny = nx.to(device), ny.to(device)

    # Pack the training data
    training_data = (
        (x_inner, y_inner),
        (x_bd, y_bd),
        (x_inner_v, y_inner_v),
        (x_bd_v, y_bd_v),
        (nx, ny),
    )

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

    prev_value_v = dict()
    prev_value_v["u0"] = exact_sol(x_inner_v, y_inner_v, 0.0 * Dt, Re, "u")
    prev_value_v["v0"] = exact_sol(x_inner_v, y_inner_v, 0.0 * Dt, Re, "v")
    prev_value_v["p0"] = exact_sol(x_inner_v, y_inner_v, 0.0 * Dt, Re, "p")
    prev_value_v["u1"] = exact_sol(x_inner_v, y_inner_v, 1.0 * Dt, Re, "u")
    prev_value_v["v1"] = exact_sol(x_inner_v, y_inner_v, 1.0 * Dt, Re, "v")
    prev_value_v["p1"] = exact_sol(x_inner_v, y_inner_v, 1.0 * Dt, Re, "p")
    prev_value_v["du0dx"], prev_value_v["du0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 0.0 * Dt, Re, "u")
    prev_value_v["dv0dx"], prev_value_v["dv0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 0.0 * Dt, Re, "v")
    prev_value_v["du1dx"], prev_value_v["du1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 1.0 * Dt, Re, "u")
    prev_value_v["dv1dx"], prev_value_v["dv1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 1.0 * Dt, Re, "v")
    prev_value_v["dp0dx"], prev_value_v["dp0dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 0.0 * Dt, Re, "p")
    prev_value_v["dp1dx"], prev_value_v["dp1dy"] = vmap(
        grad(exact_sol, argnums=(0, 1)), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_inner_v.reshape(-1), y_inner_v.reshape(-1), 1.0 * Dt, Re, "p")

    # reshape the values to (n, 1)
    for key, key_v in iter(zip(prev_value, prev_value_v)):
        prev_value[key] = prev_value[key].reshape(-1, 1)
        prev_value_v[key_v] = prev_value_v[key_v].reshape(-1, 1)
        # print(f'prev_value["{key}"]: {prev_value[key].size()}')
        # print(f'prev_value_v["{key_v}"]: {prev_value_v[key_v].size()}')
    
    # Predict the intermediate velocity field (u*, v*)
    prediction_step(model, training_data, prev_value, prev_value_v)

    


if __name__ == "__main__":
    Re = 400
    Dt = 0.01
    main()