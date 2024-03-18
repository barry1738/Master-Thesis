import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.tensor([1, 1, -1])

dist = torch.where(z > 0, x, y)
print(dist)