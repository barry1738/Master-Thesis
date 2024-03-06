import torch
from torch.func import jacrev, vmap

torch.set_default_dtype(torch.float64)


def f(coord):
    x = coord[:, 0]
    y = coord[:, 1]
    func = torch.sin(x) * torch.sin(y)
    return func.reshape(-1, 1)


x = torch.randint(1, 10, (5, 2), dtype=torch.float64)

jac = jacrev(f)(x).squeeze()
grad = torch.cos(x[:, 0]) * torch.sin(x[:, 1])

jac2 = vmap(jacrev(jacrev(f)), in_dims=0, out_dims=0)(x).squeeze()
# grad2 = -torch.sin(x)

print(f'x = \n{x}')
print(f'jac = \n{jac}')
print(f'grad = \n{grad}')

print(f'jac2 = \n{jac2}')
print(f'grad2 = \n{grad2}')