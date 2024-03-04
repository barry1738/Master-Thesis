import torch

list1 = []

a = torch.rand(2, 1)
b = torch.rand(2, 1)

list1.append(a)
print(list1)
print(len(list1))

list1.append(None)
print(list1)
print(len(list1))