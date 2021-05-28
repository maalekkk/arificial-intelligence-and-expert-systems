import numpy as np
import pandas as pd
from path import Path
from torch import nn, optim, randperm, tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_lr_finder import LRFinder

from constants import F8_TEST_FILES_PATHS, F10_TEST_FILES_PATHS, COLUMNS
from animation import plot_track, calc_and_plot_distribution
from datasetUwb import Dataset, get_data_from_excel
from errors import calc_error


def save_to_excel(distribution, file_name):
    df = pd.DataFrame(distribution, columns=['Distribution'])
    df.to_excel('./distributions/distribution_' + file_name + '.xlsx', index=False)


class NeuralNetwork(nn.Module):
    def __init__(self, inputs_size, output_size, hidden_neurons=None):
        super(NeuralNetwork, self).__init__()
        if hidden_neurons is None:
            hidden_neurons = [64, 32, 16]
        assert (len(hidden_neurons) == 3)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs_size, hidden_neurons[0]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[1], hidden_neurons[2]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[2], output_size),
            nn.ReLU()
        )

    def forward(self, data):
        data = self.flatten(data)
        vector = self.linear_relu_stack(data)
        return vector

    def perform_training(self, epochs, training_data, reference_data, lr=2.07E-03, shuffle=True):
        assert (epochs > 0)
        assert (len(training_data) == len(reference_data))
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=5e-5)
        return_loss = 0
        for epoch in range(epochs):
            return_loss = 0.0
            if shuffle:
                idx = randperm(training_data.nelement())
                training_data = training_data.view(-1)[idx].view(training_data.size())
                reference_data = reference_data.view(-1)[idx].view(reference_data.size())
            dataset = TensorDataset(training_data, reference_data)
            dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
            for data, ref_data in dataloader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, ref_data)
                loss.backward()
                optimizer.step()
                return_loss += loss.item()
            print('[ Epoch ' + str(epoch + 1) + ' ] : ' + str(return_loss / len(dataloader)))
        return return_loss

    def predict(self, data):
        output = self(data)
        return output.detach().numpy()

    def find_lr(self, data, ref_data):
        dataset = TensorDataset(data, ref_data)
        dataloader = DataLoader(dataset)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.000001, weight_decay=5e-5)
        lr_finder = LRFinder(self, optimizer, criterion)
        lr_finder.range_test(dataloader, end_lr=1, num_iter=500)
        lr_finder.plot()
        lr_finder.reset()


def predict_calc_and_plot(file_path):
    name = Path(file_path).name[:-5]
    data, ref_data = get_data_from_excel(file_path)

    prediction = network.predict(tensor(data))
    network_error = calc_error(ref_data=ref_data, output=prediction)
    reference_error = np.sum(np.abs(data - ref_data), axis=1)

    plot_track(coords_test=data, reference_test=ref_data, output=prediction, filename=name)
    output_cdf = calc_and_plot_distribution(distribution_org=reference_error, distribution=network_error,
                                            filename=name)
    save_to_excel(output_cdf, name)


if __name__ == '__main__':
    # Import train data
    train_data = Dataset(10)
    coords_train, reference_train = train_data.import_static_data()

    # Create model of neural network
    network = NeuralNetwork(2, 2)

    # Find learning rate
    # network.find_lr(coords_train, reference_train)

    # Training
    network.perform_training(200, coords_train, reference_train)

    # Predict, calculate error and plot for F8 audience
    # for path in F8_TEST_FILES_PATHS:
    #     predict_calc_and_plot(path)

    # Predict, calculate error and plot for F10 audience
    for path in F10_TEST_FILES_PATHS:
        predict_calc_and_plot(path)
