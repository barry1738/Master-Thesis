import torch
import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick

torch.set_default_dtype(torch.float64)
plt.rcParams.update({"font.size": 12})


def exact_solution(x, y):
    return torch.sin(3 * torch.pi * x) * torch.sin(3 * torch.pi * y) 

def rhs_function(x, y):
    u_xx = torch.vmap(
        torch.func.grad(torch.func.grad(exact_solution, argnums=0), argnums=0),
        in_dims=(0, 0),
        out_dims=0,
    )(x.reshape(-1), y.reshape(-1))
    u_yy = torch.vmap(
        torch.func.grad(torch.func.grad(exact_solution, argnums=1), argnums=1),
        in_dims=(0, 0),
        out_dims=0,
    )(x.reshape(-1), y.reshape(-1))
    return (u_xx + u_yy).reshape(-1, 1)


class Model:
    def __init__(self, M_p, X_min, X_max, Y_min, Y_max):
        self.M_p = M_p
        self.X_min = X_min
        self.X_max = X_max
        self.Y_min = Y_min
        self.Y_max = Y_max

    def mapping_x(self, x, idx):
        xn = (self.X_max - self.X_min) * (2 * (idx + 1) - 1) / (2 * self.M_p)
        rn = (self.X_max - self.X_min) / (2 * self.M_p)
        return (x - xn) / rn
    
    def mapping_y(self, y, idx):
        yn = (self.Y_max - self.Y_min) * (2 * (idx + 1) - 1) / (2 * self.M_p)
        rn = (self.Y_max - self.Y_min) / (2 * self.M_p)
        return (y - yn) / rn
    
    def PoU(self, x):
        return torch.where((x >= -1) & (x <= 1), 1.0, 0.0)
    
    def predict(self, weights, biases, x, y, i, j):
        input = torch.hstack((self.mapping_x(x, i), self.mapping_y(y, j)))
        y = torch.nn.Sigmoid()(torch.nn.functional.linear(input, weights, biases))
        return y
    
    def predict_dx(self, weights, biases, x, y, i, j):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict(weights, biases, primal, y, i, j), x
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_dy(self, weights, biases, x, y, i, j):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict(weights, biases, x, primal, i, j), y
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_dxx(self, weights, biases, x, y, i, j):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict_dx(weights, biases, primal, y, i, j), x
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_dyy(self, weights, biases, x, y, i, j):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict_dy(weights, biases, x, primal, i, j), y
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_func(self, weights, biases, points, i, j):
        func = torch.vmap(
            self.predict, in_dims=(2, 1, None, None, None, None), out_dims=1
        )(
            weights[:, :, :, i], biases[:, :, i], points[i][j][0], points[i][j][1], i, j
        ).squeeze()
        return func
    
    def predict_dx_func(self, weights, biases, points, i, j):
        func = torch.vmap(
            self.predict_dx, in_dims=(2, 1, None, None, None, None), out_dims=1
        )(
            weights[:, :, :, i], biases[:, :, i], points[i][j][0], points[i][j][1], i, j
        ).squeeze()
        return func
    
    def predict_dy_func(self, weights, biases, points, i, j):
        func = torch.vmap(
            self.predict_dy, in_dims=(2, 1, None, None, None, None), out_dims=1
        )(
            weights[:, :, :, i], biases[:, :, i], points[i][j][0], points[i][j][1], i, j
        ).squeeze()
        return func
    
    def predict_dxx_func(self, weights, biases, points, i, j):
        func = torch.vmap(
            self.predict_dxx, in_dims=(2, 1, None, None, None, None), out_dims=1
        )(
            weights[:, :, :, i], biases[:, :, i], points[i][j][0], points[i][j][1], i, j
        ).squeeze()
        return func
    
    def predict_dyy_func(self, weights, biases, points, i, j):
        func = torch.vmap(
            self.predict_dyy, in_dims=(2, 1, None, None, None, None), out_dims=1
        )(
            weights[:, :, :, i], biases[:, :, i], points[i][j][0], points[i][j][1], i, j
        ).squeeze()
        return func

    def assemble_matrix(self, weights, biases, points, J_n, Q):
        (
            internal_points,
            boundary_points_left,
            boundary_points_right,
            boundary_points_bottom,
            boundary_points_top,
        ) = points

        A_I = torch.zeros(self.M_p**2 * Q**2, self.M_p**2 * J_n)  # PDE term
        A_B_1 = torch.zeros(self.M_p * Q, self.M_p**2 * J_n) # left boundary term
        A_B_2 = torch.zeros(self.M_p * Q, self.M_p**2 * J_n) # right boundary term
        A_B_3 = torch.zeros(self.M_p * Q, self.M_p**2 * J_n) # bottom boundary term
        A_B_4 = torch.zeros(self.M_p * Q, self.M_p**2 * J_n) # top boundary term
        A_X_C_0 = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 0-order smoothness term x direction
        A_X_C_1_dx = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 1-order smoothness term x direction
        A_X_C_1_dy = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 1-order smoothness term x direction
        A_Y_C_0 = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 0-order smoothness term y direction
        A_Y_C_1_dx = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 1-order smoothness term y direction
        A_Y_C_1_dy = torch.zeros((self.M_p - 1) * self.M_p * Q, self.M_p**2 * J_n)  # 1-order smoothness term y direction
        f_I = torch.zeros(self.M_p**2 * Q**2, 1)  # PDE right-hand side
        f_1 = torch.zeros(self.M_p * Q, 1)  # left boundary right-hand side
        f_2 = torch.zeros(self.M_p * Q, 1)  # right boundary right-hand side
        f_3 = torch.zeros(self.M_p * Q, 1)  # bottom boundary right-hand side
        f_4 = torch.zeros(self.M_p * Q, 1)  # top boundary right-hand side
        f_x_c_0 = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 0-order smoothness x right-hand side
        f_x_c_1_dx = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 1-order smoothness x right-hand side
        f_x_c_1_dy = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 1-order smoothness x right-hand side
        f_y_c_0 = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 0-order smoothness y right-hand side
        f_y_c_1_dx = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 1-order smoothness y right-hand side
        f_y_c_1_dy = torch.zeros((self.M_p - 1) * self.M_p * Q, 1)  # 1-order smoothness y right-hand side

        # print(f'A_I: {A_I.shape}')
        # print(f'A_B: {A_B.shape}')
        # print(f'A_X_C_0: {A_X_C_0.shape}')
        # print(f'A_X_C_1: {A_X_C_1.shape}')
        # print(f'A_Y_C_0: {A_Y_C_0.shape}')
        # print(f'A_Y_C_1: {A_Y_C_1.shape}')

        for i in range(self.M_p):
            for j in range(self.M_p):
                # print(f'PoU region: {i * self.M_p + j}, i = {i}, j = {j}')
                Q_begin = (i * self.M_p + j) * Q**2
                J_n_begin = (i * self.M_p + j) * J_n

                A_predict_dxx = self.predict_dxx_func(weights, biases, internal_points, i, j)
                
                A_predict_dyy = self.predict_dyy_func(weights, biases, internal_points, i, j)

                # Assemble the PDE term
                A_I[Q_begin : Q_begin + Q**2, J_n_begin : J_n_begin + J_n] = (
                    A_predict_dxx + A_predict_dyy
                )

                # Assemble the PDE term right-hand side
                f_I[Q_begin : Q_begin + Q**2, :] = (
                    rhs_function(internal_points[i][j][0], internal_points[i][j][1])
                )

                # Assemble the boundary term
                if i == 0:  # left boundary
                    A_B_1[j * Q : (j + 1) * Q, J_n_begin : J_n_begin + J_n] = (
                        self.predict_func(weights, biases, boundary_points_left, i, j)
                    )
                    f_1[j * Q : (j + 1) * Q, :] = exact_solution(
                        boundary_points_left[i][j][0], boundary_points_left[i][j][1]
                    )

                if i == self.M_p - 1:  # right boundary
                    A_B_2[j * Q : (j + 1) * Q, J_n_begin : J_n_begin + J_n] = (
                        self.predict_func(weights, biases, boundary_points_right, i, j)
                    )
                    f_2[j * Q : (j + 1) * Q, :] = exact_solution(
                        boundary_points_right[i][j][0], boundary_points_right[i][j][1]
                    )

                if j == 0:  # bottom boundary
                    A_B_3[i * Q : (i + 1) * Q, J_n_begin : J_n_begin + J_n] = (
                        self.predict_func(weights, biases, boundary_points_bottom, i, j)
                    )
                    f_3[i * Q : (i + 1) * Q, :] = exact_solution(
                        boundary_points_bottom[i][j][0], boundary_points_bottom[i][j][1]
                    )

                if j == self.M_p - 1:  # top boundary
                    A_B_4[i * Q : (i + 1) * Q, J_n_begin : J_n_begin + J_n] = (
                        self.predict_func(weights, biases, boundary_points_top, i, j)
                    )
                    f_4[i * Q : (i + 1) * Q, :] = exact_solution(
                        boundary_points_top[i][j][0], boundary_points_top[i][j][1]
                    )

                # x-axis smoothness term
                if self.M_p > 1:
                    x_idx_begin = j * (self.M_p - 1) * Q + i * Q
                    if i == 0:
                        A_X_C_0[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_func(weights, biases, boundary_points_right, i, j)
                        )
                        A_X_C_1_dx[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dx_func(weights, biases, boundary_points_right, i, j)
                        )
                        A_X_C_1_dy[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dy_func(weights, biases, boundary_points_right, i, j)
                        )
                    elif i == self.M_p - 1:
                        A_X_C_0[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_func(weights, biases, boundary_points_left, i, j)
                        )
                        A_X_C_1_dx[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dx_func(weights, biases, boundary_points_left, i, j)
                        )
                        A_X_C_1_dy[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dy_func(weights, biases, boundary_points_left, i, j)
                        )
                    else:
                        A_X_C_0[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_func(weights, biases, boundary_points_right, i, j)
                        )
                        A_X_C_0[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_func(weights, biases, boundary_points_left, i, j)
                        )
                        A_X_C_1_dx[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dx_func(weights, biases, boundary_points_right, i, j)
                        )
                        A_X_C_1_dx[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dx_func(weights, biases, boundary_points_left, i, j)
                        )
                        A_X_C_1_dy[x_idx_begin : x_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dy_func(weights, biases, boundary_points_right, i, j)
                        )
                        A_X_C_1_dy[x_idx_begin - Q : x_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dy_func(weights, biases, boundary_points_left, i, j)
                        )

                # y-axis smoothness term
                if self.M_p > 1:
                    y_idx_begin = i * (self.M_p - 1) * Q + j * Q
                    if j == 0:
                        A_Y_C_0[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_func(weights, biases, boundary_points_top, i, j)
                        )
                        A_Y_C_1_dx[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dx_func(weights, biases, boundary_points_top, i, j)
                        )
                        A_Y_C_1_dy[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dy_func(weights, biases, boundary_points_top, i, j)
                        )
                    elif j == self.M_p - 1:
                        A_Y_C_0[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_func(weights, biases, boundary_points_bottom, i, j)
                        )
                        A_Y_C_1_dx[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dx_func(weights, biases, boundary_points_bottom, i, j)
                        )
                        A_Y_C_1_dy[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dy_func(weights, biases, boundary_points_bottom, i, j)
                        )
                    else:
                        A_Y_C_0[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_func(weights, biases, boundary_points_top, i, j)
                        )
                        A_Y_C_0[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_func(weights, biases, boundary_points_bottom, i, j)
                        )
                        A_Y_C_1_dx[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dx_func(weights, biases, boundary_points_top, i, j)
                        )
                        A_Y_C_1_dx[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dx_func(weights, biases, boundary_points_bottom, i, j)
                        )
                        A_Y_C_1_dy[y_idx_begin : y_idx_begin + Q, J_n_begin : J_n_begin + J_n] = (
                            self.predict_dy_func(weights, biases, boundary_points_top, i, j)
                        )
                        A_Y_C_1_dy[y_idx_begin - Q : y_idx_begin, J_n_begin : J_n_begin + J_n] = (
                            - self.predict_dy_func(weights, biases, boundary_points_bottom, i, j)
                        )

        if self.M_p > 1:
            A = torch.vstack((A_I, A_B_1, A_B_2, A_B_3, A_B_4, A_X_C_0, A_Y_C_0, A_X_C_1_dx, A_Y_C_1_dx, A_X_C_1_dy, A_Y_C_1_dy))
            f = torch.vstack((f_I, f_1, f_2, f_3, f_4, f_x_c_0, f_y_c_0, f_x_c_1_dx, f_y_c_1_dx, f_x_c_1_dy, f_y_c_1_dy))
        else:
            A = torch.vstack((A_I, A_B_1, A_B_2, A_B_3, A_B_4))
            f = torch.vstack((f_I, f_1, f_2, f_3, f_4))

        print(f'A: {A.shape}')
        print(f'f: {f.shape}')
        return A, f
    

def test_model(model, weights, biases, J_n, Q, u_hat):
    exact_values = []
    numerical_values = []
    error = []
    plot_points_x = []
    plot_points_y = []

    for i in range(model.M_p):
        x_min = model.X_min + (model.X_max - model.X_min) * i / model.M_p
        x_max = model.X_min + (model.X_max - model.X_min) * (i + 1) / model.M_p
        for j in range(model.M_p):
            y_min = model.Y_min + (model.Y_max - model.Y_min) * j / model.M_p
            y_max = model.Y_min + (model.Y_max - model.Y_min) * (j + 1) / model.M_p

            # test_x = torch.Tensor(5 * Q * Q, 1, device=torch.device("cpu")).uniform_(x_min, x_max)
            # test_y = torch.Tensor(5 * Q * Q, 1, device=torch.device("cpu")).uniform_(y_min, y_max)
            test_x, test_y = torch.meshgrid(
                torch.linspace(x_min, x_max, 5 * Q),
                torch.linspace(y_min, y_max, 5 * Q),
                indexing="xy",
            )
            test_x = test_x.reshape(-1, 1)
            test_y = test_y.reshape(-1, 1)

            matrix = torch.vmap(
                model.predict, in_dims=(2, 1, None, None, None, None), out_dims=1
            )(weights[:, :, :, i], biases[:, :, i], test_x, test_y, i, j).squeeze()

            exact_value = exact_solution(test_x, test_y)
            numerical_value = (
                matrix @ u_hat[(i * model.M_p + j) * J_n : (i * model.M_p + j + 1) * J_n, :]
            )
            exact_values.extend(exact_value)
            numerical_values.extend(numerical_value)
            error.extend(torch.abs(exact_value - numerical_value))
            plot_points_x.extend(test_x)
            plot_points_y.extend(test_y)

    plot_points_x = torch.tensor(plot_points_x).reshape(-1, 1)
    plot_points_y = torch.tensor(plot_points_y).reshape(-1, 1)
    exact_values = torch.tensor(exact_values).reshape(-1, 1)
    numerical_values = torch.tensor(numerical_values).reshape(-1, 1)
    error = torch.tensor(error).reshape(-1, 1)
    print(f'max error = {torch.max(error)}')

    # fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].scatter(plot_points, exact_values, label="Exact solution", color="red")
    # 3D plot
    sca1 = axs[0].scatter(plot_points_x, plot_points_y, numerical_values,
                          c=numerical_values, label="Numerical solution", marker=".")
    sca2 = axs[1].scatter(plot_points_x, plot_points_y, error,
                          c=error, label="Error", marker=".")
    # 2D plot
    sca1 = axs[0].scatter(plot_points_x, plot_points_y,
                          c=numerical_values, label="Numerical solution", marker=".")
    sca2 = axs[1].scatter(plot_points_x, plot_points_y,
                          c=error, label="Error", marker=".")
    # axs[0].legend()
    # axs[1].legend()
    axs[0].set_title("Numerical solution")
    axs[1].set_title("Error")
    axs[0].axis('square')
    axs[1].axis('square')
    # axs[1].zaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
    # axs[1].tick_params(axis="z", pad=15)
    plt.colorbar(sca1, shrink=0.7, aspect=15)
    plt.colorbar(sca2, shrink=0.7, aspect=15)
    plt.suptitle(r"u = sin(3$\pi$x)sin(3$\pi$y)")
    plt.show()
    # plt.savefig("poisson_2d.png")

    return exact_values, numerical_values, error


def main():
    J_n = 200  # the number of basis functions per PoU region
    Q = 10  # the number of collocation points per axis of PoU region
    M_p = 3  # the number of PoU regions per axis
    R_m = 1  # the range of the random weights and biases

    # Initialize the weights and biases
    weights = torch.randn(1, 2, J_n, M_p**2).uniform_(-R_m, R_m)
    biases = torch.randn(1, J_n, M_p**2).uniform_(-R_m, R_m)

    # Create the training and tset points data
    X_min = torch.tensor(0.0)
    X_max = torch.tensor(1.0)
    Y_min = torch.tensor(0.0)
    Y_max = torch.tensor(1.0)
    internal_points = []
    boundary_points_left = []
    boundary_points_right = []
    boundary_points_top = []
    boundary_points_bottom = []
    for i in range(M_p):
        internal_p = []
        boundary_p_left = []
        boundary_p_right = []
        boundary_p_top = []
        boundary_p_bottom = []

        x_min = X_min + (X_max - X_min) * i / M_p
        x_max = X_min + (X_max - X_min) * (i + 1) / M_p
        for j in range(M_p):
            y_min = Y_min + (Y_max - Y_min) * j / M_p
            y_max = Y_min + (Y_max - Y_min) * (j + 1) / M_p

            internal_x = torch.Tensor(Q**2, 1).uniform_(x_min, x_max)
            internal_y = torch.Tensor(Q**2, 1).uniform_(y_min, y_max)

            boundary_x_left = x_min * torch.ones(Q, 1)
            # boundary_y_left = torch.vstack(
            #     (y_min, (torch.Tensor(Q - 2, 1).uniform_(y_min, y_max)), y_max)
            # )
            boundary_y_left = torch.torch.linspace(y_min, y_max, Q).reshape(-1, 1)

            boundary_x_right = x_max * torch.ones(Q, 1)
            # boundary_y_right = torch.vstack(
            #     (y_min, (torch.Tensor(Q - 2, 1).uniform_(y_min, y_max)), y_max)
            # )
            boundary_y_right = torch.torch.linspace(y_min, y_max, Q).reshape(-1, 1)

            # boundary_x_top = torch.vstack(
            #     (x_min, (torch.Tensor(Q - 2, 1).uniform_(x_min, x_max)), x_max)
            # )
            boundary_x_top = torch.torch.linspace(x_min, x_max, Q).reshape(-1, 1)
            boundary_y_top = y_max * torch.ones(Q, 1)

            # boundary_x_bottom = torch.vstack(
            #     (x_min, (torch.Tensor(Q - 2, 1).uniform_(x_min, x_max)), x_max)
            # )
            boundary_x_bottom = torch.torch.linspace(x_min, x_max, Q).reshape(-1, 1)
            boundary_y_bottom = y_min * torch.ones(Q, 1)

            # append the points
            internal_p.append([internal_x, internal_y])
            boundary_p_left.append([boundary_x_left, boundary_y_left])
            boundary_p_right.append([boundary_x_right, boundary_y_right])
            boundary_p_top.append([boundary_x_top, boundary_y_top])
            boundary_p_bottom.append([boundary_x_bottom, boundary_y_bottom])

        # append the points
        internal_points.append(internal_p)
        boundary_points_left.append(boundary_p_left)
        boundary_points_right.append(boundary_p_right)
        boundary_points_top.append(boundary_p_top)
        boundary_points_bottom.append(boundary_p_bottom)

    # Plot the points
    plot_intermal_x = []
    plot_intermal_y = []
    plot_boundary_x = []
    plot_boundary_y = []
    plot_jump_x = []
    plot_jump_y = []
    for i in range(M_p):
        for j in range(M_p):
            plot_intermal_x.extend(internal_points[i][j][0])
            plot_intermal_y.extend(internal_points[i][j][1])

            if i == 0:
                plot_boundary_x.extend(boundary_points_left[i][j][0])
                plot_boundary_y.extend(boundary_points_left[i][j][1])
            else:
                plot_jump_x.extend(boundary_points_left[i][j][0])
                plot_jump_y.extend(boundary_points_left[i][j][1])

            if i == M_p - 1:
                plot_boundary_x.extend(boundary_points_right[i][j][0])
                plot_boundary_y.extend(boundary_points_right[i][j][1])
            else:
                plot_jump_x.extend(boundary_points_right[i][j][0])
                plot_jump_y.extend(boundary_points_right[i][j][1])

            if j == 0:
                plot_boundary_x.extend(boundary_points_bottom[i][j][0])
                plot_boundary_y.extend(boundary_points_bottom[i][j][1])
            else:
                plot_jump_x.extend(boundary_points_bottom[i][j][0])
                plot_jump_y.extend(boundary_points_bottom[i][j][1])

            if j == M_p - 1:
                plot_boundary_x.extend(boundary_points_top[i][j][0])
                plot_boundary_y.extend(boundary_points_top[i][j][1])
            else:
                plot_jump_x.extend(boundary_points_top[i][j][0])
                plot_jump_y.extend(boundary_points_top[i][j][1])

    fig, ax = plt.subplots()
    ax.scatter(plot_intermal_x, plot_intermal_y, label="Internal points", color="blue", marker=".")
    ax.scatter(plot_boundary_x, plot_boundary_y, label="Boundary points", color="red", marker=".")
    ax.scatter(plot_jump_x, plot_jump_y, label="Jump points", color="green", marker=".")
    ax.legend()
    ax.axis('square')
    plt.show()

    # Create the model
    model = Model(M_p, X_min, X_max, Y_min, Y_max)
    # Assemble the linear system
    A, f = model.assemble_matrix(
        weights,
        biases,
        (
            internal_points,
            boundary_points_left,
            boundary_points_right,
            boundary_points_bottom,
            boundary_points_top,
        ),
        J_n,
        Q,
    )

    # Solve the linear system using least squares
    # u_hat = torch.linalg.lstsq(A, f).solution

    # Solve the linear system using SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u_hat = Vh.t() @ (torch.diag_embed(1.0 / S) @ (U.t() @ f))

    # Solve the linear system using QR decomposition
    # Q, R = torch.linalg.qr(A)
    # u_hat = torch.linalg.solve_triangular(R, Q.t() @ f, upper=True)
    print(f"u_hat: {u_hat.shape}")

    # Test the model
    test_model(model, weights, biases, J_n, Q, u_hat)


if __name__ == "__main__":
    main()
