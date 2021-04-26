from torch import nn, optim

import DatasetUWB


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 2)

    # def __init__(self):
    #     super(NeuralNetwork, self).__init__()
    #     self.flatten = nn.Flatten()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(2, 10),
    #         nn.ReLU(),
    #         nn.Linear(10, 10),
    #         nn.ReLU(),
    #         nn.Linear(10, 2),
    #         nn.ReLU()
    #     )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    # def forward(self, x):
    #     x = self.flatten(x)
    #     logits = self.linear_relu_stack(x)
    #     return logits

    def perform_training(self, epochs, training_data, reference_data):
        assert(epochs > 0)
        assert(len(training_data) == len(reference_data))

        criterion = nn.L1Loss()
        optimizer = optim.SGD(self.parameters(), lr=0.05, momentum=0.05)

        for epoch in range(epochs):
            running_loss = 0

            for _ in range(len(training_data)):
                optimizer.zero_grad()
                output = self(training_data)
                loss = criterion(output, reference_data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(running_loss / len(training_data))


if __name__ == '__main__':
    dataset = DatasetUWB.DatasetUWD(8)
    dataset.import_static_data()
    c, r = dataset.get_torch_dataset()
    model = NeuralNetwork()

    print(model)
    model.perform_training(100, c, r)

