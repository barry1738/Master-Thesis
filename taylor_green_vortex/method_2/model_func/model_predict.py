import torch
from torch.func import functional_call, vjp

def predict(model, params, x, y):
    """Predict the model output."""
    return functional_call(model, params, (x, y))


def predict_dx(model, params, x, y):
    """Compute the directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (primal, y)), x)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dy(model, params, x, y):
    """Compute the directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: functional_call(model, params, (x, primal)), y)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dxx(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to x."""
    output, vjpfunc = vjp(lambda primal: predict_dx(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dxy(model, params, x, y):
    """Compute the mixed directional derivative of the model output with respect to x and y."""
    output, vjpfunc = vjp(lambda primal: predict_dx(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dyx(model, params, x, y):
    """Compute the mixed directional derivative of the model output with respect to y and x."""
    output, vjpfunc = vjp(lambda primal: predict_dy(model, params, primal, y), x)
    return vjpfunc(torch.ones_like(output))[0]


def predict_dyy(model, params, x, y):
    """Compute the second directional derivative of the model output with respect to y."""
    output, vjpfunc = vjp(lambda primal: predict_dy(model, params, x, primal), y)
    return vjpfunc(torch.ones_like(output))[0]