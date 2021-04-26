from torch import nn, optim
import Errors
import Animation
import DatasetUWB


class NeuralNetwork(nn.Module):

    def __init__(self, inputs_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs_size, 100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        vector = self.linear_relu_stack(x)
        return vector

    def perform_training(self, epochs, training_data, reference_data):
        assert (epochs > 0)
        assert (len(training_data) == len(reference_data))
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        for epoch in range(epochs):
            running_loss = 0
            optimizer.zero_grad()
            output = self(training_data)
            loss = criterion(output, reference_data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(running_loss / len(training_data))

    def predict(self, data):
        output = self(data)
        return output.detach().numpy()


if __name__ == '__main__':
    train_data = DatasetUWB.DatasetUWD(8)
    train_data.import_static_data()
    coords_train, reference_train = train_data.get_torch_dataset()

    test_data = DatasetUWB.DatasetUWD(8)
    test_data.import_file("./dane/pomiary/F8/f8_random_2p.xlsx")
    coords_test, reference_test = test_data.get_torch_dataset()

    network = NeuralNetwork(2, 2)
    network.perform_training(1000, coords_train, reference_train)

    out = network.predict(coords_test)

    dist = Errors.calc_distribution(reference_test, out)
    dist_org = Errors.calc_distribution(reference_test, coords_test)

    Animation.distribution_plot(dist, dist_org, 'distribution_f8.png')

    Animation.track_plot(coords_test, reference_test, out, 'track.png')
