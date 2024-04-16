import torch
import model_func as mf
from projection_module.config import REYNOLDS_NUM as Re

def update_step(model, params, points, prev_value, prev_value_valid, device):
    """The update step for velocity and pressure fields"""
    # Unpack the model
    u_star_model = model[0]
    v_star_model = model[1]
    phi_model = model[2]
    psi_model = model[3]

    # Unpack the parameters
    u_star_params = params[0]
    v_star_params = params[1]
    phi_params = params[2]
    psi_params = params[3]

    # Unpack the training data
    x_training = points[0]
    y_training = points[1]
    x_test = points[2]
    y_test = points[3]

    # Move the data to the device
    x_training, y_training = x_training.to(device), y_training.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Create the new dictionary for the update value
    new_value = dict()
    new_value_valid = dict()
    new_value["u0"] = prev_value["u1"]
    new_value["v0"] = prev_value["v1"]
    new_value["p0"] = prev_value["p1"]
    new_value["du0dx"] = prev_value["du1dx"]
    new_value["du0dy"] = prev_value["du1dy"]
    new_value["dv0dx"] = prev_value["dv1dx"]
    new_value["dv0dy"] = prev_value["dv1dy"]
    new_value["dp0dx"] = prev_value["dp1dx"]
    new_value["dp0dy"] = prev_value["dp1dy"]

    new_value_valid["u0"] = prev_value_valid["u1"]
    new_value_valid["v0"] = prev_value_valid["v1"]
    new_value_valid["p0"] = prev_value_valid["p1"]
    new_value_valid["du0dx"] = prev_value_valid["du1dx"]
    new_value_valid["du0dy"] = prev_value_valid["du1dy"]
    new_value_valid["dv0dx"] = prev_value_valid["dv1dx"]
    new_value_valid["dv0dy"] = prev_value_valid["dv1dy"]
    new_value_valid["dp0dx"] = prev_value_valid["dp1dx"]
    new_value_valid["dp0dy"] = prev_value_valid["dp1dy"]

    new_value["u1"] = mf.predict_dy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["v1"] = -mf.predict_dx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["p1"] = (
        torch.tensor(prev_value["p1"], device=device)
        + mf.predict(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            mf.predict_dx(u_star_model, u_star_params, x_training, y_training)
            + mf.predict_dy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    new_value["du1dx"] = mf.predict_dyx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["du1dy"] = mf.predict_dyy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dv1dx"] = -mf.predict_dxx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dv1dy"] = -mf.predict_dxy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dp1dx"] = (
        torch.tensor(prev_value["dp1dx"], device=device)
        + mf.predict_dx(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            mf.predict_dxx(u_star_model, u_star_params, x_training, y_training)
            + mf.predict_dyx(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    new_value["dp1dy"] = (
        torch.tensor(prev_value["dp1dy"], device=device)
        + mf.predict_dy(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            mf.predict_dxy(u_star_model, u_star_params, x_training, y_training)
            + mf.predict_dyy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()

    new_value_valid["u1"] = mf.predict_dy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["v1"] = -mf.predict_dx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["p1"] = (
        torch.tensor(prev_value_valid["p1"], device=device)
        + mf.predict(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            mf.predict_dx(u_star_model, u_star_params, x_test, y_test)
            + mf.predict_dy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    new_value_valid["du1dx"] = mf.predict_dyx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["du1dy"] = mf.predict_dyy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dv1dx"] = -mf.predict_dxx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dv1dy"] = -mf.predict_dxy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dp1dx"] = (
        torch.tensor(prev_value_valid["dp1dx"], device=device)
        + mf.predict_dx(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            mf.predict_dxx(u_star_model, u_star_params, x_test, y_test)
            + mf.predict_dyx(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    new_value_valid["dp1dy"] = (
        torch.tensor(prev_value_valid["dp1dy"], device=device)
        + mf.predict_dy(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            mf.predict_dxy(u_star_model, u_star_params, x_test, y_test)
            + mf.predict_dyy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()

    return new_value, new_value_valid