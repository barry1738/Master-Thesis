import torch
from torch.utils.data import Dataset, DataLoader

# Creating our dataset class
class Build_Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.func = -5 * self.x + 1
        self.y = self.func + 0.4 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting length of the data
    def __len__(self):
        return self.len

data = Build_Data()
print(data.__len__())