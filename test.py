import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def weights_init(self, model):
        if isinstance(model, nn.Linear):
            # nn.init.xavier_uniform_(model.weight.data)
            model.reset_parameters()

    def forward(self, x, y):
        """Forward pass of the neural network."""
        input = torch.hstack((x, y))
        for i in range(len(self.linear_layers) - 1):
            input = self.activation(self.linear_layers[i](input))
        output = self.linear_layers[-1](input)
        return output
    

model = Model([2, 5, 1])
params = dict(model.named_parameters())
for key, value in params.items():
    print(key, value)

print("\n\n")

model.apply(model.weights_init)
params2 = dict(model.named_parameters())
for key, value in params2.items():
    print(key, value)