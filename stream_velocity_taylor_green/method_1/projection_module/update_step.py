def update_step(training_data, u_star_model, v_star_model, phi_model, psi_model,
                u_star_params, v_star_params, phi_params, psi_params, prev_value, 
                prev_value_valid):
    """The update step for velocity and pressure fields"""
    # Unpack the training data
    x_inner, y_inner = training_data[0]
    x_bd, y_bd = training_data[1]
    x_inner_v, y_inner_v = training_data[2]
    x_bd_v, y_bd_v = training_data[3]

    x_training = torch.vstack((x_inner, x_bd))
    y_training = torch.vstack((y_inner, y_bd))
    x_test = torch.vstack((x_inner_v, x_bd_v))
    y_test = torch.vstack((y_inner_v, y_bd_v))

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

    # new_value["u1"] = (
    #     u_star_model.predict(u_star_model, u_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dx(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    # new_value["v1"] = (
    #     v_star_model.predict(v_star_model, v_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dy(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    new_value["u1"] = psi_model.predict_dy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["v1"] = -psi_model.predict_dx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["p1"] = (
        torch.tensor(prev_value["p1"], device=device)
        + phi_model.predict(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dx(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    # new_value["du1dx"] = (
    #     u_star_model.predict_dx(u_star_model, u_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dxx(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    # new_value["du1dy"] = (
    #     u_star_model.predict_dy(u_star_model, u_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dxy(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    # new_value["dv1dx"] = (
    #     v_star_model.predict_dx(v_star_model, v_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dyx(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    # new_value["dv1dy"] = (
    #     v_star_model.predict_dy(v_star_model, v_star_params, x_training, y_training)
    #     - (2 * Dt / 3) * phi_model.predict_dyy(phi_model, phi_params, x_training, y_training)
    # ).cpu().detach().numpy()
    new_value["du1dx"] = psi_model.predict_dyx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["du1dy"] = psi_model.predict_dyy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dv1dx"] = -psi_model.predict_dxx(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dv1dy"] = -psi_model.predict_dxy(psi_model, psi_params, x_training, y_training).cpu().detach().numpy()
    new_value["dp1dx"] = (
        torch.tensor(prev_value["dp1dx"], device=device)
        + phi_model.predict_dx(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dxx(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dyx(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()
    new_value["dp1dy"] = (
        torch.tensor(prev_value["dp1dy"], device=device)
        + phi_model.predict_dy(phi_model, phi_params, x_training, y_training)
        - (1 / Re) * (
            phi_model.predict_dxy(u_star_model, u_star_params, x_training, y_training)
            + phi_model.predict_dyy(v_star_model, v_star_params, x_training, y_training)
        )
    ).cpu().detach().numpy()

    # new_value_valid["u1"] = (
    #     u_star_model.predict(u_star_model, u_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dx(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    # new_value_valid["v1"] = (
    #     v_star_model.predict(v_star_model, v_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dy(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    new_value_valid["u1"] = psi_model.predict_dy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["v1"] = -psi_model.predict_dx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["p1"] = (
        torch.tensor(prev_value_valid["p1"], device=device)
        + phi_model.predict(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dx(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    # new_value_valid["du1dx"] = (
    #     u_star_model.predict_dx(u_star_model, u_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dxx(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    # new_value_valid["du1dy"] = (
    #     u_star_model.predict_dy(u_star_model, u_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dxy(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    # new_value_valid["dv1dx"] = (
    #     v_star_model.predict_dx(v_star_model, v_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dyx(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    # new_value_valid["dv1dy"] = (
    #     v_star_model.predict_dy(v_star_model, v_star_params, x_test, y_test)
    #     - (2 * Dt / 3) * phi_model.predict_dyy(phi_model, phi_params, x_test, y_test)
    # ).cpu().detach().numpy()
    new_value_valid["du1dx"] = psi_model.predict_dyx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["du1dy"] = psi_model.predict_dyy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dv1dx"] = -psi_model.predict_dxx(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dv1dy"] = -psi_model.predict_dxy(psi_model, psi_params, x_test, y_test).cpu().detach().numpy()
    new_value_valid["dp1dx"] = (
        torch.tensor(prev_value_valid["dp1dx"], device=device)
        + phi_model.predict_dx(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dxx(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dyx(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()
    new_value_valid["dp1dy"] = (
        torch.tensor(prev_value_valid["dp1dy"], device=device)
        + phi_model.predict_dy(phi_model, phi_params, x_test, y_test)
        - (1 / Re) * (
            phi_model.predict_dxy(u_star_model, u_star_params, x_test, y_test)
            + phi_model.predict_dyy(v_star_model, v_star_params, x_test, y_test)
        )
    ).cpu().detach().numpy()

    return new_value, new_value_valid