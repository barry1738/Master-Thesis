import torch
from torch.func import jacrev, vmap

torch.set_default_dtype(torch.float64)

a = torch.tensor([[1.0]])
print(f'a: {a.size()}')