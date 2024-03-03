import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


def exact_solution(x):
    func = (
        torch.sin(3.0 * torch.pi * (x + 0.05)) * 
        torch.cos(2.0 * torch.pi * (x + 0.05))
        + 2.0
    )
    return func


def rhs_function(x):
    u_xx = torch.vmap(torch.func.grad(torch.func.grad(
        exact_solution, argnums=0), argnums=0),
        in_dims=0, out_dims=0)(x.reshape(-1))
    return u_xx.reshape(-1, 1) - 4 * exact_solution(x)


class Model:
    def __init__(self, M_p):
        self.M_p = M_p

    def mapping(self, x, idx):
        xn = 8 * (2 * (idx + 1) - 1) / (2 * self.M_p)
        rn = 8 / (2 * self.M_p)
        return (x - xn) / rn
    
    def PoU(self, x):
        return torch.where((x >= -1) & (x <= 1), 1.0, 0.0)
    
    def predict(self, weights, biases, x, idx):
        # psi = self.PoU(self.mapping(x, idx))
        y = torch.nn.Sigmoid()(
            torch.nn.functional.linear(self.mapping(x, idx), weights, biases)
        )
        return y
    
    def predict_dx(self, weights, biases, x, idx):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict(weights, biases, primal, idx), x
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def predict_dxx(self, weights, biases, x, idx):
        output, vjpfunc = torch.func.vjp(
            lambda primal: self.predict_dx(weights, biases, primal, idx), x
        )
        return vjpfunc(torch.ones_like(output))[0]
    
    def assemble_matrix(self, weights, biases, points, J_n, Q):
        A_I = torch.zeros(self.M_p * Q, self.M_p * J_n)  # PDE term
        A_B = torch.zeros(2, self.M_p * J_n) # Boundary term
        A_C_0 = torch.zeros(self.M_p - 1, self.M_p * J_n)  # 0-order smoothness term
        A_C_1 = torch.zeros(self.M_p - 1, self.M_p * J_n)  # 1-order smoothness term
        f = torch.zeros(self.M_p * Q + 2 * (self.M_p - 1) + 2, 1)  # Right-hand side

        for i in range(self.M_p):
            values = torch.vmap(self.predict, in_dims=(2, 1, None, None), out_dims=1)(
                weights[:, :, :, i], biases[:, :, i], points[i], i
            ).squeeze()
            value_l, value_r = values[0, :], values[-1, :]

            grad1 = torch.vmap(self.predict_dx, in_dims=(2, 1, None, None), out_dims=1)(
                weights[:, :, :, i], biases[:, :, i], points[i], i
            ).squeeze()
            grad1_l, grad1_r = grad1[0, :], grad1[-1, :]

            grad2 = torch.vmap(self.predict_dxx, in_dims=(2, 1, None, None), out_dims=1)(
                weights[:, :, :, i], biases[:, :, i], points[i], i
            ).squeeze()

            # print(f'values: {values.shape}')
            # print(f'grad1: {grad1.shape}')
            # print(f'grad2: {grad2.shape}')

            # fig, ax = plt.subplots()
            # ax.scatter(points[i], self.mapping(points[i], i))
            # plt.show()

            Lu = grad2 - 4 * values

            # Assemble the PDE term
            A_I[i * Q : (i + 1) * Q, i * J_n : (i + 1) * J_n] = Lu[1:-1, :]
            f[i * Q : (i + 1) * Q, :] = rhs_function(points[i])[1:-1, :]
        
            # Boundary conditions
            if i == 0:
                A_B[0, :J_n] = value_l
            if i == self.M_p - 1:
                A_B[1, -J_n:] = value_r

            # Smoothness conditions
            if self.M_p > 0:
                if i == 0:
                    A_C_0[0, :J_n] = -value_r
                    A_C_1[0, :J_n] = -grad1_r
                elif i == self.M_p - 1:
                    A_C_0[-1, -J_n:] = value_l
                    A_C_1[-1, -J_n:] = grad1_l
                else:
                    A_C_0[i - 1, i * J_n : (i + 1) * J_n] = value_l
                    A_C_0[i, i * J_n : (i + 1) * J_n] = -value_r
                    A_C_1[i - 1, i * J_n : (i + 1) * J_n] = grad1_l
                    A_C_1[i, i * J_n : (i + 1) * J_n] = -grad1_r

        if self.M_p > 1:
            A = torch.vstack((A_I, A_B, A_C_0, A_C_1))
        else:
            A = torch.vstack((A_I, A_B))

        # RHS boundary conditions
        f[self.M_p * Q, :] = exact_solution(torch.tensor(0.0))
        f[self.M_p * Q + 1, :] = exact_solution(torch.tensor(8.0))

        # print(f'A: {A.shape}')
        # print(f'f: {f.shape}')

        return A, f
    

def test_model(model, weights, biases, points, J_n, Q, u_hat):
    exact_values = []
    numerical_values = []
    error = []
    plot_points = []

    for i in range(model.M_p):
        values = torch.vmap(model.predict, in_dims=(2, 1, None, None), out_dims=1)(
            weights[:, :, :, i], biases[:, :, i], points[i], i
        ).squeeze()

        exact_value = exact_solution(points[i])
        numerical_value = values @ u_hat[i * J_n : (i + 1) * J_n, :]
        exact_values.extend(exact_value)
        numerical_values.extend(numerical_value)
        error.extend(torch.abs(exact_value - numerical_value))
        plot_points.extend(points[i])

    plot_points = torch.tensor(plot_points).reshape(-1, 1)
    exact_values = torch.tensor(exact_values).reshape(-1, 1)
    numerical_values = torch.tensor(numerical_values).reshape(-1, 1)

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(plot_points, exact_values, label="Exact solution", color="red", marker="o")
    axs[0].scatter(plot_points, numerical_values, label="Numerical solution", color="blue", marker=".")
    axs[1].scatter(plot_points, error, label="Error", color="green", marker=".")
    axs[0].legend()
    axs[1].legend()
    plt.show()

    return exact_values, numerical_values, error


def main():
    J_n = 100  # the number of basis functions per PoU region
    Q = 100  # the number of collocation pointss per PoU region
    M_p = 8  # the number of PoU regions
    R_m = 3  # the range of the random weights and biases

    # Initialize the weights and biases
    weights = torch.randn(1, 1, J_n, M_p).uniform_(-R_m, R_m)
    biases = torch.randn(1, J_n, M_p).uniform_(-R_m, R_m)

    # Create the training and tset points data
    X_min = torch.tensor(0.0)
    X_max = torch.tensor(8.0)
    training_points = []
    test_points = []
    for i in range(M_p):
        x_min = X_min + (X_max - X_min) * i / M_p
        x_max = X_min + (X_max - X_min) * (i + 1) / M_p
        x_training = torch.Tensor(Q, 1).uniform_(x_min, x_max)
        x_test = torch.Tensor(5 * Q, 1).uniform_(x_min, x_max)  # 5 times more test points
        training_points.append(torch.vstack((x_min, x_training, x_max)))
        test_points.append(torch.vstack((x_min, x_test, x_max)))

    # Create the model
    model = Model(M_p)
    # Assemble the linear system
    A, f = model.assemble_matrix(weights, biases, training_points, J_n, Q)

    # Solve the linear system
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u_hat = Vh.t() @ (torch.diag_embed(1.0 / S) @ (U.t() @ f))

    # Test the model
    test_model(model, weights, biases, test_points, J_n, Q, u_hat)


if __name__ == "__main__":
    main()
