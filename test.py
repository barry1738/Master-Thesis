import torch
from tensordict import TensorDict

a = torch.rand(3, 4)
b = torch.rand(3, 4, 5)
tensordict = TensorDict({}, batch_size=[])
tensordict['a'] = a
tensordict['b'] = b
for key, value in tensordict.items():
    print(key, value.size())