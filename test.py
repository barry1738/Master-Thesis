import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.func import functional_call, vjp, jacrev, vmap


torch.set_default_dtype(torch.float64)

# torch.set_num_threads(1)
num_threads = torch.get_num_threads()
print(f"Benchmarking on {num_threads} threads")


class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    # def weights_init(self, model):
    #     if isinstance(model, nn.Linear):
    #         # nn.init.xavier_uniform_(model.weight.data)
    #         model.reset_parameters()

    def forward(self, x, y):
        """Forward pass of the neural network."""
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output
    
    def forward_dx(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(
            lambda primal: functional_call(model, params, (primal, y)), x
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_dy(self, model, params, x, y):
        """Compute the directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(
            lambda primal: functional_call(model, params, (x, primal)), y
        )
        return vjpfunc(torch.ones_like(output))[0]

    def forward_dxx(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to x."""
        output, vjpfunc = vjp(lambda primal: self.forward_dx(model, params, primal, y), x)
        return vjpfunc(torch.ones_like(output))[0]


    def forward_dyy(self, model, params, x, y):
        """Compute the second directional derivative of the model output with respect to y."""
        output, vjpfunc = vjp(lambda primal: self.forward_dy(model, params, x, primal), y)
        return vjpfunc(torch.ones_like(output))[0]
    

def finite_diff(model, params, x, y):
    pred_xx = (
        functional_call(model, params, (x+dx, y))
        - 2*functional_call(model, params, (x, y))
        + functional_call(model, params, (x-dx, y))
    ) / dx**2
    pred_yy = (
        functional_call(model, params, (x, y+dx))
        - 2*functional_call(model, params, (x, y))
        + functional_call(model, params, (x, y-dx))
    ) / dx**2
    return pred_xx + pred_yy


def auto_diff(model, params, x, y):
    pred_xx = model.forward_dxx(model, params, x, y)
    pred_yy = model.forward_dyy(model, params, x, y)
    return pred_xx + pred_yy


def compute_jacobian_finitie_diff(model, params, x, y):
    jacobian_dict = vmap(
        jacrev(finite_diff, argnums=1), in_dims=(None, None, 0, 0), out_dims=0
    )(model, params, x, y)

    jacobian = torch.hstack([val.view(x.size(0), -1) for val in jacobian_dict.values()])
    return jacobian


def compute_jacobian_auto_diff(model, params, x, y):
    jacobian_dict = vmap(
        jacrev(auto_diff, argnums=1), in_dims=(None, None, 0, 0), out_dims=0
    )(model, params, x, y)

    jacobian = torch.hstack([val.view(x.size(0), -1) for val in jacobian_dict.values()])
    return jacobian


def main():
    model = Model([2, 100, 100, 1])
    params = dict(model.named_parameters())
    params_flat = 10 * nn.utils.parameters_to_vector(params.values())
    # nn.utils.vector_to_parameters(params_flat, params.values())

    nx = 5000
    points_x = torch.rand(nx, 1)
    points_y = torch.rand(nx, 1)

    print(f'nrows = {nx}, ncols = {params_flat.numel()}')

    # print(functional_call(model, params, (points_x, points_y)))

    # jacobian_finite_diff = compute_jacobian_finitie_diff(
    #     model, params, points_x, points_y
    # )
    # jacobian_auto_diff = compute_jacobian_auto_diff(
    #     model, params, points_x, points_y
    # )
    # print(jacobian_finite_diff)
    # print(jacobian_auto_diff)

    # Benchmark finite difference
    timer_finite_diff = benchmark.Timer(
        stmt="compute_jacobian_finitie_diff(m, p, x, y)",
        setup="from __main__ import compute_jacobian_finitie_diff",
        globals={'m': model, 'p': params, 'x': points_x, 'y': points_y},
        num_threads=num_threads,
    )
    timer_auto_diff = benchmark.Timer(
        stmt="compute_jacobian_auto_diff(m, p, x, y)",
        setup="from __main__ import compute_jacobian_auto_diff",
        globals={"m": model, "p": params, "x": points_x, "y": points_y},
        num_threads=num_threads,
    )

    print(timer_finite_diff.timeit(10))
    print(timer_auto_diff.timeit(10))


if __name__ == "__main__":
    dx = 1e-4
    main()