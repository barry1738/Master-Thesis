"""
Taylor-Green Vortex flow using Physics-Informed Neural Networks (PINNs)
Stream function version

Incompressible Navier-Stokes Equations:
u_tt + u*u_x - v*u_y = - p_x - RE*Δu
v_tt + u*v_x + v*v_y = - p_y - RE*Δv
u_x + v_y = 0

Stream function:
ψ_y = u, ψ_x = -v

Projection Method:
Step 1: Predict the intermediate velocity field (u*, v*) using the neural network
Step 2: Project the intermediate velocity field onto the space of divergence-free fields
Step 3: Update the velocity and pressure fields
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import functional_call, vmap, jacrev, vjp, grad
import model_func as mf
import projeciton_module as pm

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
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    

def plot_loss_figure(training_loss, test_loss, title, file_name):
    """Plot the loss function"""
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogy(training_loss, "k-", label="training loss")
    ax.semilogy(test_loss, "r--", label="test loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    # fig.tight_layout()
    plt.savefig(pwd + dir_name + "figures\\loss\\" + file_name, dpi=150)
    plt.close()


def plot_figure(x_data, y_data, plot_val, title, file_name):
    """Plot the figure"""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout="constrained")
    ax.scatter(x_data, y_data, plot_val, c=plot_val, cmap="coolwarm", s=5)
    # fig.colorbar(sca, shrink=0.5, aspect=10, pad=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.zaxis.set_tick_params(pad=5)
    # ax.axes.zaxis.set_ticklabels([])
    ax.set_title(title)
    plt.savefig(pwd + dir_name + "figures\\" + file_name, dpi=150)
    plt.close()


def main():
    # Define the neural network
    u_star_model = PinnModel([2, 20, 20, 1]).to(device)
    v_star_model = PinnModel([2, 20, 20, 1]).to(device)
    phi_model = PinnModel([2, 20, 20, 1]).to(device)
    psi_model = PinnModel([2, 20, 20, 1]).to(device)
    u_model = PinnModel([2, 20, 20, 1]).to(device)
    v_model = PinnModel([2, 20, 20, 1]).to(device)
    p_model = PinnModel([2, 20, 20, 1]).to(device)

    torch.save(u_star_model, pwd + dir_name + "models\\u_star_model.pt")
    torch.save(v_star_model, pwd + dir_name + "models\\v_star_model.pt")
    torch.save(phi_model, pwd + dir_name + "models\\phi_model.pt")
    torch.save(psi_model, pwd + dir_name + "models\\psi_model.pt")
    torch.save(u_model, pwd + dir_name + "models\\u_model.pt")
    torch.save(v_model, pwd + dir_name + "models\\v_model.pt")
    torch.save(p_model, pwd + dir_name + "models\\p_model.pt")

    # Initialize the weights of the neural network
    u_star_params = u_star_model.state_dict()
    v_star_params = v_star_model.state_dict()
    phi_params = phi_model.state_dict()
    psi_params = psi_model.state_dict()
    u_params = u_model.state_dict()
    v_params = v_model.state_dict()
    p_params = p_model.state_dict()
    u_params_old = u_model.state_dict().copy()
    v_params_old = v_model.state_dict().copy()
    p_params_old = p_model.state_dict().copy()

    total_params_u_star = u_star_model.num_total_params()
    total_params_v_star = v_star_model.num_total_params()
    total_params_phi = phi_model.num_total_params()
    total_params_psi = psi_model.num_total_params()
    print(f"Total number of u* parameters: {total_params_u_star}")
    print(f"Total number of v* parameters: {total_params_v_star}")
    print(f"Total number of phi parameters: {total_params_phi}")
    print(f"Total number of psi parameters: {total_params_psi}")

    # Define the training data
    mesh = pm.CreateSquareMesh()
    x_inner, y_inner = mesh.inner_points(1000)
    x_bd, y_bd = mesh.boundary_points(30)
    x_inner_valid, y_inner_valid = mesh.inner_points(90000)
    x_bd_valid, y_bd_valid = mesh.boundary_points(300)

    # Plot the training data
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(x_inner, y_inner, s=5, c="k", label="Inner points")
    ax.scatter(x_bd, y_bd, s=5, c="r", label="Boundary points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Training data")
    ax.set_aspect("equal")
    # ax.legend()
    plt.savefig(pwd + dir_name + "figures\\training_data.png", dpi=150)

    # Move the data to the device
    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
    x_inner_valid, y_inner_valid = x_inner_valid.to(device), y_inner_valid.to(device)
    x_bd_valid, y_bd_valid = x_bd_valid.to(device), y_bd_valid.to(device)

    points = (x_inner, y_inner, x_bd, y_bd, x_inner_valid, y_inner_valid, x_bd_valid, y_bd_valid)

    for step in range(2, int(time_end / Dt) + 1):
        print(f"Step {step}, time = {Dt * step:.3f} ...")
        print("=====================================")
        # # Predict the intermediate velocity field (u*, v*)
        # Compute the right-hand side of the prediction step
        Rf_u_star, Rf_v_star, Rf_u_star_bd, Rf_v_star_bd = pm.prediction_rhs(
            (u_model, v_model, p_model), 
            (u_params, v_params, p_params, u_params_old, v_params_old, p_params_old),
            (x_inner, x_bd), (y_inner, y_bd),
            step, device
        )
        Rf_u_star_valid, Rf_v_star_valid, Rf_u_star_bd_valid, Rf_v_star_bd_valid = pm.prediction_rhs(
            (u_model, v_model, p_model), 
            (u_params, v_params, p_params, u_params_old, v_params_old, p_params_old),
            (x_inner_valid, x_bd_valid), (y_inner_valid, y_bd_valid),
            step, device
        )
        # Start training u* and v*
        u_star_params, loss_u_star, loss_u_star_valid = pm.prediction_step(
            u_star_model, 
            points,
            (Rf_u_star, Rf_u_star_bd, Rf_u_star_valid, Rf_u_star_bd_valid),
            device
        )
        v_star_params, loss_v_star, loss_v_star_valid = pm.prediction_step(
            v_star_model, 
            points,
            (Rf_v_star, Rf_v_star_bd, Rf_v_star_valid, Rf_v_star_bd_valid),
            device
        )
        # Plot the loss function
        plot_loss_figure(loss_u_star, loss_u_star_valid, "Loss of u*", f"loss_u_star_{step}.png")
        plot_loss_figure(loss_v_star, loss_v_star_valid, "Loss of v*", f"loss_v_star_{step}.png")
        # Save the parameters
        torch.save(u_star_params, pwd + dir_name + f"u_star_params\\u_star_{step}.pt")
        torch.save(v_star_params, pwd + dir_name + f"v_star_params\\v_star_{step}.pt")
        print("Finish the prediction step ...\n")

        # # Project the intermediate velocity field onto the space of divergence-free fields
        # Compute the right-hand side of the projection step
        Rf_proj_1 = mf.predict(u_star_model, u_star_params, x_inner, y_inner)
        Rf_proj_2 = mf.predict(v_star_model, v_star_params, x_inner, y_inner)
        Rf_proj_bd_1 = pm.exact_sol(x_bd, y_bd, step * Dt, Re, "u")
        Rf_proj_bd_2 = pm.exact_sol(x_bd, y_bd, step * Dt, Re, "v")
        Rf_proj_1_valid = mf.predict(u_star_model, u_star_params, x_inner_valid, y_inner_valid)
        Rf_proj_2_valid = mf.predict(v_star_model, v_star_params, x_inner_valid, y_inner_valid)
        Rf_proj_bd_1_valid = pm.exact_sol(x_bd_valid, y_bd_valid, step * Dt, Re, "u")
        Rf_proj_bd_2_valid = pm.exact_sol(x_bd_valid, y_bd_valid, step * Dt, Re, "v")
        phi_params, psi_params, loss_proj, loss_proj_valid = pm.projection_step(
            phi_model, psi_model, points, 
            (Rf_proj_1, Rf_proj_2, Rf_proj_bd_1, Rf_proj_bd_2, Rf_proj_1_valid, 
             Rf_proj_2_valid, Rf_proj_bd_1_valid, Rf_proj_bd_2_valid),
            device
        )
        # Plot the loss function
        plot_loss_figure(loss_proj, loss_proj_valid, "Loss of projection step", f"loss_proj_{step}.png")
        # Save the parameters
        torch.save(phi_params, pwd + dir_name + f"phi_params\\phi_{step}.pt")
        torch.save(psi_params, pwd + dir_name + f"psi_params\\psi_{step}.pt")
        print("Finish the projection step ...\n")

        # # Update the velocity field and pressure field
        # Compute the right-hand side of the update step
        x = torch.vstack((x_inner, x_bd))
        y = torch.vstack((y_inner, y_bd))
        x_v = torch.vstack((x_inner_valid, x_bd_valid))
        y_v = torch.vstack((y_inner_valid, y_bd_valid))

        Rf_u = mf.predict_dy(psi_model, psi_params, x, y)
        Rf_v = -mf.predict_dx(psi_model, psi_params, x, y)
        if step == 2:
            Rf_p = (
                pm.exact_sol(x, y, (step - 1) * Dt, Re, "p")
                + mf.predict(phi_model, phi_params, x, y)
                - (1 / Re) * (
                    mf.predict_dx(u_star_model, u_star_params, x, y)
                    + mf.predict_dy(v_star_model, v_star_params, x, y)
                )
            )
        else:
            Rf_p = (
                mf.predict(p_model, p_params_old, x, y)
                + mf.predict(phi_model, phi_params, x, y)
                - (1 / Re) * (
                    mf.predict_dx(u_star_model, u_star_params, x, y)
                    + mf.predict_dy(v_star_model, v_star_params, x, y)
                )
            )

        # Update the velocity parameters and pressure parameters
        u_params_old = u_params.copy()
        v_params_old = v_params.copy()
        p_params_old = p_params.copy()

        # Compute the new velocity parameters and pressure parameters
        u_params = pm.update_step(u_model, points, Rf_u, device)
        v_params = pm.update_step(v_model, points, Rf_v, device)
        p_params = pm.update_step(p_model, points, Rf_p, device)

        torch.save(u_params, pwd + dir_name + f"u_params\\u_{step}.pt")
        torch.save(v_params, pwd + dir_name + f"v_params\\v_{step}.pt")
        torch.save(p_params, pwd + dir_name + f"p_params\\p_{step}.pt")
        print("Finish the update step ...\n")

        # if step % 10 == 0:
        #     # Plot the velocity field and pressure field
        #     exact_u = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "u").detach().numpy()
        #     exact_v = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "v").detach().numpy()
        #     exact_p = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "p").detach().numpy()
        #     error_u = np.abs(prev_value_valid["u1"] - exact_u)
        #     error_v = np.abs(prev_value_valid["v1"] - exact_v)
        #     error_p = np.abs(prev_value_valid["p1"] - exact_p)
        #     print('Final compute the error ...\n')

        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=prev_value_valid["u1"],
        #         title="Predicted u",
        #         file_name=f"u\\u_{step}.png",
        #     )
        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=prev_value_valid["v1"],
        #         title="Predicted v",
        #         file_name=f"v\\v_{step}.png",
        #     )
        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=prev_value_valid["p1"],
        #         title="Predicted p",
        #         file_name=f"p\\p_{step}.png",
        #     )
        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=error_u,
        #         title="Error of u",
        #         file_name=f"u\\u_error_{step}.png",
        #     )
        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=error_v,
        #         title="Error of v",
        #         file_name=f"v\\v_error_{step}.png",
        #     )
        #     plot_figure(
        #         x_data=prev_value_valid["x_data"],
        #         y_data=prev_value_valid["y_data"],
        #         plot_val=error_p,
        #         title="Error of p",
        #         file_name=f"p\\p_error_{step}.png",
        #     )
        #     print('Finish the plot ...\n')


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Set the font size in figures
    plt.rcParams.update({"font.size": 12})

    Re = pm.REYNOLDS_NUM
    Dt = pm.TIME_STEP
    time_end = 0.02

    print(f'Re = {Re}, Dt = {Dt}, time_end = {time_end} ...')

    pwd = "C:\\barry_doc\\Training_Data\\"
    dir_name = "TaylorGreenVortex_Streamfunction_1000_0.05\\"
    if not os.path.exists(pwd + dir_name):
        print("Creating data directory...")
        os.makedirs(pwd + dir_name)
        os.makedirs(pwd + dir_name + "models")
        os.makedirs(pwd + dir_name + "u_star_params")
        os.makedirs(pwd + dir_name + "v_star_params")
        os.makedirs(pwd + dir_name + "phi_params")
        os.makedirs(pwd + dir_name + "psi_params")
        os.makedirs(pwd + dir_name + "u_params")
        os.makedirs(pwd + dir_name + "v_params")
        os.makedirs(pwd + dir_name + "p_params")
        os.makedirs(pwd + dir_name + "figures\\u")
        os.makedirs(pwd + dir_name + "figures\\v")
        os.makedirs(pwd + dir_name + "figures\\p")
        os.makedirs(pwd + dir_name + "figures\\loss")
    else:
        print("Data directory already exists...")

    main()
