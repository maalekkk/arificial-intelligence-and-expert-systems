import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x
