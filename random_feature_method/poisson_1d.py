import torch

class Model:
    def __init__(self, Mp):
        self.Mp = Mp

    def mapping(self, x, idx):
        xn = (2 * (idx + 1) - 1) / (2 * self.Mp)
        rn = 1 / (2 * self.Mp)
        return (x - xn) / rn
    
    def PoU(self, x):
        return torch.where(x >= -1 & x <= 1, 1.0, 0.0)
    
    def predict(self, weights, biases, x, idx):
        psi = self.PoU(self.mapping(x, idx))
        y = torch.nn.Sigmoid()(
            torch.nn.functional.linear(self.mapping(x, idx), weights, biases)
        )
        return psi * y
    
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
    
    def assemble_matrix(self, points)


def main():
    J_n = 50  # the number of basis functions per PoU region
    Q = 50  # the number of collocation pointss per PoU region
    M_p = 5  # the number of PoU regions

    # Initialize the weights and biases
    weights = torch.randn(1, 1, J_n, M_p)
    biases = torch.randn(1, J_n, M_p)

    # Create the training and tset points data
    x_training = torch.linspace(0, 1, Q * M_p + 1).reshape(-1, 1)
    x_test = torch.linspace(0, 1, 100 * M_p + 1).reshape(-1, 1)
    print(x_training.shape)

    model = Model(M_p)




if __name__ == "__main__":
    main()
