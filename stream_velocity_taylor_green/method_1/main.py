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
import model_func as mf
import projection_module as pm


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


def weights_init(model):
    """Initialize the weights of the neural network."""
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data, gain=5)
        # nn.init.xavier_normal_(model.weight.data, gain=10)


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

    torch.save(u_star_model, pwd + dir_name + "models\\u_star_model.pt")
    torch.save(v_star_model, pwd + dir_name + "models\\v_star_model.pt")
    torch.save(phi_model, pwd + dir_name + "models\\phi_model.pt")
    torch.save(psi_model, pwd + dir_name + "models\\psi_model.pt")

    total_params_u_star = u_star_model.num_total_params()
    total_params_v_star = v_star_model.num_total_params()
    total_params_phi = phi_model.num_total_params()
    total_params_psi = psi_model.num_total_params()
    print(f"Total number of u* parameters: {total_params_u_star}")
    print(f"Total number of v* parameters: {total_params_v_star}")
    print(f"Total number of phi parameters: {total_params_phi}")
    print(f"Total number of psi parameters: {total_params_psi}")

    # Define the training data
    mesh = mf.CreateSquareMesh()
    x_inner, y_inner = mesh.inner_points(1000)
    x_bd, y_bd = mesh.boundary_points(30)
    x_inner_valid, y_inner_valid = mesh.inner_points(90000)
    x_bd_valid, y_bd_valid = mesh.boundary_points(300)

    # Pack the training data
    training_data = (
        (x_inner, y_inner),
        (x_bd, y_bd),
        (x_inner_valid, y_inner_valid),
        (x_bd_valid, y_bd_valid),
    )

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

    # Initialize the previous value
    x_training = torch.vstack((x_inner, x_bd))
    y_training = torch.vstack((y_inner, y_bd))
    x_test = torch.vstack((x_inner_valid, x_bd_valid))
    y_test = torch.vstack((y_inner_valid, y_bd_valid))

    prev_value, prev_value_valid = pm.initial_value(
        x_training, y_training, x_test, y_test, Dt, Re
    )
    # Save the initial value
    torch.save(prev_value, pwd + dir_name + "data\\data_1.pt")
    torch.save(prev_value_valid, pwd + dir_name + "data\\data_valid_1.pt")
    
    for step in range(2, int(time_end / Dt) + 1):
        print(f"Step {step}, time = {Dt * step:.3f} ...")
        print("=====================================")

        # # Predict the intermediate velocity field (u*, v*)
        print("===== Start the prediction step ... =====")
        # Compute the right-hand side of the Navier-Stokes equations
        Rf_u_inner, Rf_v_inner, Rf_u_bd, Rf_v_bd = pm.prediction_step_rhs(
            (x_inner, x_bd), (y_inner, y_bd), prev_value, step, Dt, Re, device
        )
        Rf_u_inner_valid, Rf_v_inner_valid, Rf_u_bd_valid, Rf_v_bd_valid = pm.prediction_step_rhs(
            (x_inner_valid, x_bd_valid), (y_inner_valid, y_bd_valid), 
            prev_value_valid, step, Dt, Re, device
        )

        # Train the neural network
        while True:
            u_star_params, loss_u_star, loss_u_star_valid = pm.prediction_step(
                u_star_model, 
                (x_inner, y_inner, x_bd, y_bd, x_inner_valid, y_inner_valid, x_bd_valid, y_bd_valid), 
                (Rf_u_inner, Rf_u_bd, Rf_u_inner_valid, Rf_u_bd_valid), device
            )
            if loss_u_star_valid[-1] / loss_u_star[-1] > 5.0:
                print("Re-training the u* model ...")
            elif loss_u_star[-1] / loss_u_star_valid[-1] > 5.0:
                print("Re-training the u* model ...")
            else:
                break

        while True:
            v_star_params, loss_v_star, loss_v_star_valid = pm.prediction_step(
                v_star_model, 
                (x_inner, y_inner, x_bd, y_bd, x_inner_valid, y_inner_valid, x_bd_valid, y_bd_valid), 
                (Rf_v_inner, Rf_v_bd, Rf_v_inner_valid, Rf_v_bd_valid), device
            )
            if loss_v_star_valid[-1] / loss_v_star[-1] > 5.0:
                print("Re-training the v* model ...")
            elif loss_v_star[-1] / loss_v_star_valid[-1] > 5.0:
                print("Re-training the v* model ...")
            else:
                break

        # Plot the loss function
        plot_loss_figure(
            loss_u_star,
            loss_u_star_valid,
            f"Loss function of u*, Time = {step * Dt}",
            f"u_satr\\u_star_loss_{step}.png",
        )
        plot_loss_figure(
            loss_v_star,
            loss_v_star_valid,
            f"Loss function of v*, Time = {step * Dt}",
            f"v_star\\v_star_loss_{step}.png",
        )
        # Save the parameters
        torch.save(u_star_params, pwd + dir_name + f"u_star_model\\u_star_{step}.pt")
        torch.save(v_star_params, pwd + dir_name + f"v_star_model\\v_star_{step}.pt")
        print("Finish the prediction step ...\n")

        # # Project the intermediate velocity field onto the space of divergence-free fields
        print("===== Start the projection step ... =====")
        # Compute the right-hand side of the Poisson equation
        Rf_1 = u_star_model.predict(u_star_model, u_star_params, x_inner, y_inner)
        Rf_2 = v_star_model.predict(v_star_model, v_star_params, x_inner, y_inner)
        Rf_1_valid = u_star_model.predict(u_star_model, u_star_params, x_inner_valid, y_inner_valid)
        Rf_2_valid = v_star_model.predict(v_star_model, v_star_params, x_inner_valid, y_inner_valid)
        Rf_1_bd = mf.exact_sol(x_bd, y_bd, step * Dt, Re, "u")
        Rf_2_bd = mf.exact_sol(x_bd, y_bd, step * Dt, Re, "v")
        Rf_1_bd_valid = mf.exact_sol(x_bd_valid, y_bd_valid, step * Dt, Re, "u")
        Rf_2_bd_valid = mf.exact_sol(x_bd_valid, y_bd_valid, step * Dt, Re, "v")

        # Train the neural network
        while True:
            phi_params, psi_params = mf.projection_step(
                u_star_model, v_star_model, phi_model, psi_model, u_star_params, v_star_params, step
            )

        # phi_model.load_state_dict(phi_params)
        # psi_model.load_state_dict(psi_params)
        torch.save(phi_model, pwd + dir_name + f"phi_model\\phi_{step}.pt")
        torch.save(psi_model, pwd + dir_name + f"psi_model\\psi_{step}.pt")
        print("Finish the projection step ...\n")

        # Update the velocity field and pressure field
        new_value, new_value_valid = update_step(
            training_data, u_star_model, v_star_model, phi_model, psi_model,
            u_star_params, v_star_params, phi_params, psi_params,
            prev_value, prev_value_valid
        )

        prev_value.update(new_value)
        prev_value_valid.update(new_value_valid)
        torch.save(prev_value, pwd + dir_name + f"data\\data_{step}.pt")
        torch.save(prev_value_valid, pwd + dir_name + f"data\\data_valid_{step}.pt")
        print("Finish the update step ...\n")

        if step % 10 == 0:
            # Plot the velocity field and pressure field
            exact_u = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "u").detach().numpy()
            exact_v = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "v").detach().numpy()
            exact_p = exact_sol(torch.tensor(prev_value_valid["x_data"]), torch.tensor(prev_value_valid["y_data"]), step * Dt, Re, "p").detach().numpy()
            error_u = np.abs(prev_value_valid["u1"] - exact_u)
            error_v = np.abs(prev_value_valid["v1"] - exact_v)
            error_p = np.abs(prev_value_valid["p1"] - exact_p)
            print('Final compute the error ...\n')

            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=prev_value_valid["u1"],
                title="Predicted u",
                file_name=f"u\\u_{step}.png",
            )
            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=prev_value_valid["v1"],
                title="Predicted v",
                file_name=f"v\\v_{step}.png",
            )
            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=prev_value_valid["p1"],
                title="Predicted p",
                file_name=f"p\\p_{step}.png",
            )
            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=error_u,
                title="Error of u",
                file_name=f"u\\u_error_{step}.png",
            )
            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=error_v,
                title="Error of v",
                file_name=f"v\\v_error_{step}.png",
            )
            plot_figure(
                x_data=prev_value_valid["x_data"],
                y_data=prev_value_valid["y_data"],
                plot_val=error_p,
                title="Error of p",
                file_name=f"p\\p_error_{step}.png",
            )
            print('Finish the plot ...\n')


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Set the font size in figures
    plt.rcParams.update({"font.size": 12})

    Re = mf.REYNOLDS_NUM
    Dt = mf.TIME_STEP
    time_end = 10.0

    pwd = "C:\\barry_doc\\Training_Data\\"
    dir_name = "TaylorGreenVortex_Streamfunction_1000_0.02\\"
    if not os.path.exists(pwd + dir_name):
        print("Creating data directory...")
        os.makedirs(pwd + dir_name)
        os.makedirs(pwd + dir_name + "models")
        os.makedirs(pwd + dir_name + "params")
        os.makedirs(pwd + dir_name + "params\\u_star_model")
        os.makedirs(pwd + dir_name + "params\\v_star_model")
        os.makedirs(pwd + dir_name + "params\\phi_model")
        os.makedirs(pwd + dir_name + "params\\psi_model")
        os.makedirs(pwd + dir_name + "data")
        os.makedirs(pwd + dir_name + "figures\\u")
        os.makedirs(pwd + dir_name + "figures\\v")
        os.makedirs(pwd + dir_name + "figures\\p")
        os.makedirs(pwd + dir_name + "figures\\loss")
        os.makedirs(pwd + dir_name + "figures\\loss\\u_star")
        os.makedirs(pwd + dir_name + "figures\\loss\\v_star")
        os.makedirs(pwd + dir_name + "figures\\loss\\proj")
    else:
        print("Data directory already exists...")

    main()
