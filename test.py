import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# a = torch.tensor([[1], [2], [3]])
# b = torch.tensor([[4], [5], [6]])

print(f'a shape: {a.shape}')
print(f'b shape: {b.shape}')

# print(torch.cat((a, b), dim=1).size())
print(torch.vstack((a, b)).size())