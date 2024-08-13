"""
Learning the 1d score function via the O-U process
dXt  = -beta/2*dXt*dt + sqrt(beta)*dWt
X(0) = X0 ~ N( mu_0, sigma_0 )
--------------------------------------
score(X,t) = -( X - mu(t) )/( exp(-beta*t)*sigma_0^2 + sigma(t)^2 )
eps(X,t)   = -sigma(t)*score(X,t)
mu(t)      = exp(-beta/2*t)*mu_0
sigma(t)   = sqrt( 1 - exp(-beta*t) )
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader


torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("device = ", device)


class Build_Data(Dataset):
    # Constructor
    def __init__(self, m, T, mu_0, sigma_0, beta):
        self.m = m
        self.T = T
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.beta = beta
        self.len = m

    # Getting the data
    def __getitem__(self, index):
        return self.xin[index, :], self.yin[index], self.sigmaT[index]

    # Getting length of the data
    def __len__(self):
        return self.len
    
    def mu(self, X, t):
        return np.exp(-self.beta / 2 * t) * X

    def sigma(self, t):
        return np.sqrt(1 - np.exp(-self.beta * t))

    def score_exact(self, X, t):
        return -(X - self.mu(self.mu_0, t)) / (
            np.exp(-self.beta * t) * self.sigma_0**2 + self.sigma(t) ** 2
        )

    def eps_exact(self, X, t):
        return -self.sigma(t) * self.score_exact(X, t)
    
    def create_data(self):
        # X0 = mu_0 + sigma_0 * np.random.randn(m, 1)
        X0 = np.random.normal(self.mu_0, self.sigma_0, (self.m, 1))
        t = np.sort(np.append(self.T * np.random.rand(self.m - 1), self.T)).reshape(-1, 1)
        noise = np.random.randn(self.m, 1)
        Xt = self.mu(X0, t) + self.sigma(t) * noise
        self.sigmaT = self.sigma(t)
        # ... training points ...
        self.xin = np.hstack((Xt, t))
        # ... target output ...
        self.yin = noise


class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        # input layer
        self.ln_in = nn.Linear(in_dim, h_dim[0])

        self.hidden = nn.ModuleList()
        # hidden layers
        for i in range(len(h_dim) - 1):
            self.hidden.append(nn.Linear(h_dim[i], h_dim[i + 1]))

        # output layer
        self.ln_out = nn.Linear(h_dim[-1], out_dim, bias=True)  # bias=True or False?

        # activation function
        self.act = nn.Sigmoid()
        # self.act = nn.LogSigmoid()

    def forward(self, x):
        input = x
        input = self.act(self.ln_in(input))
        for layer in self.hidden:
            input = self.act(layer(input))
        output = self.ln_out(input)
        return output
    

def main():
    m = 1000  # number of training samples
    m_test = 1 * m  # number of testing samples
    max_iter = 1000  # maximum number of iterations

    # SDE parameters
    T = 10.0
    mu_0 = 3.0
    sigma_0 = 1.0
    beta = 3.0

    # Problem setup
    # ... Generate training data (Xt, t) ...
    # ... run SDE to generate Xt ...
    data_set = Build_Data(m, T, mu_0, sigma_0, beta)
    data_set.create_data()

    # Create model
    model = Model(2, [10], 1).to(device)
    loss_fn = nn.MSELoss(reduction="mean")

    # Calculate total number of parameters
    totWb = sum(p.numel() for p in model.parameters())

    # Move data to device
    data_set.xin = torch.tensor(data_set.xin, device=device)
    data_set.yin = torch.tensor(data_set.yin, device=device)
    data_set.sigmaT = torch.tensor(data_set.sigmaT, device=device)

    # Creating Dataloader object with batch size (number of training samples)
    trainloader = DataLoader(dataset=data_set, batch_size=int(m / 5))

    # Optimizer
    opt = "Adam"  # SGD or Adam
    if opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_SGD = []
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_Adam = []

    print(f"* Number of Trainig points: {data_set.__len__()}")
    print(f"* Total number of parameters: {totWb}")
    print(f"* Optimizer: {opt}")
    print("--------------------------------")

    # Training
    for batch_idx, (xin, yin, sigmaT) in enumerate(trainloader):
        print(f"Batch: {batch_idx + 1}")
        for epoch in range(max_iter):
            if opt == "SGD":
                # making a prediction in forward pass
                y_hat = model(xin)
                # calculating the loss between original and predicted data points
                loss = torch.sqrt(loss_fn(y_hat, -sigmaT * yin))
                # store loss into list
                loss_SGD.append(loss.item())
                # zeroing gradients after each iteration
                optimizer.zero_grad()
                # backward pass for computing the gradients of the loss w.r.t to learnable parameters
                loss.backward()
                # updating the parameters after each iteration
                optimizer.step()

            if opt == "Adam":
                # making a prediction in forward pass
                y_hat = model(xin)
                # calculating the loss between original and predicted data points
                loss = torch.sqrt(loss_fn(y_hat, -sigmaT * yin))
                # store loss into list
                loss_Adam.append(loss.item())
                # zeroing gradients after each iteration
                optimizer.zero_grad()
                # backward pass for computing the gradients of the loss w.r.t to learnable parameters
                loss.backward()
                # updating the parameters after each iteration
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"epoch: {epoch + 1}, loss: {loss.item():.4e}")

    # Plot the loss function
    print(f"Loss: {loss_SGD[-1] if opt == 'SGD' else loss_Adam[-1]: .4e}")

    fig, ax = plt.subplots()
    ax.plot(loss_SGD if opt == "SGD" else loss_Adam, label="Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

    # Testing and Output
    # ... Generate testing data (Xt, t) ...
    # ... run SDE to generate Xt ...
    data_set_test = Build_Data(m_test, T, mu_0, sigma_0, beta)
    data_set_test.create_data()
    x_test = data_set_test.xin
    y_test = model(torch.tensor(x_test, device=device)).cpu().detach().numpy()
    s_test = data_set_test.score_exact(x_test[:, 0], x_test[:, 1]).reshape(-1, 1)

    # Plot the score function
    fig, axs = plt.subplots(1, 2)
    sca1 = axs[0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=5, cmap="coolwarm")
    axs[0].set_title(r"$score_N(X_t,t)$")
    axs[0].set_ylim([0, T])
    fig.colorbar(sca1, ax=axs[0])
    sca2 = axs[1].scatter(x_test[:, 0], x_test[:, 1], c=s_test, s=5, cmap="coolwarm")
    axs[1].set_title(r"$score(X_t,t)$")
    axs[1].set_ylim([0, T])
    fig.colorbar(sca2, ax=axs[1])
    plt.show()


if __name__ == "__main__":
    main()
