from torch.func import grad, vmap
from projection_module import exact_sol


def initial_value(x_training, y_training, x_test, y_test, Dt, Re):
    prev_value = dict()
    prev_value["x_data"] = x_training.detach().numpy()
    prev_value["y_data"] = y_training.detach().numpy()
    prev_value["u0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["v0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["p0"] = exact_sol(x_training, y_training, 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["u1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["v1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["p1"] = exact_sol(x_training, y_training, 1.0 * Dt, Re, "p").detach().numpy()
    prev_value["du0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["du0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value["dv0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["dv0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value["du1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["du1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value["dv1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["dv1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value["dp0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()
    prev_value["dp1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_training.reshape(-1), y_training.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()

    prev_value_valid = dict()
    prev_value_valid["x_data"] = x_test.detach().numpy()
    prev_value_valid["y_data"] = y_test.detach().numpy()
    prev_value_valid["u0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["v0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["p0"] = exact_sol(x_test, y_test, 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["u1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["v1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["p1"] = exact_sol(x_test, y_test, 1.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["du0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["du0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["dv0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dv0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["du1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["du1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "u").detach().numpy()
    prev_value_valid["dv1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dv1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "v").detach().numpy()
    prev_value_valid["dp0dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp0dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 0.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp1dx"] = vmap(
        grad(exact_sol, argnums=0), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()
    prev_value_valid["dp1dy"] = vmap(
        grad(exact_sol, argnums=1), in_dims=(0, 0, None, None, None), out_dims=0
    )(x_test.reshape(-1), y_test.reshape(-1), 1.0 * Dt, Re, "p").detach().numpy()

    for key, key_v in iter(zip(prev_value, prev_value_valid)):
        prev_value[key] = prev_value[key].reshape(-1, 1)
        prev_value_valid[key_v] = prev_value_valid[key_v].reshape(-1, 1)

    return prev_value, prev_value_valid