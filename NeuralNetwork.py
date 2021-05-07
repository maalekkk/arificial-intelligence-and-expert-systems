import torch
from torch import nn, optim
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import Errors
import random
import Animation
import DatasetUWB
import pandas as pd


def find_optimal_layers_no(training_data, training_refdata):
    network2 = NeuralNetwork(2, 2)
    network2.linear_relu_stack = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.ReLU()
    )
    network2_err = network2.perform_training(15, training_data, training_refdata)

    network3 = NeuralNetwork(2, 2)
    network3.linear_relu_stack = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.ReLU()
    )
    network3_err = network3.perform_training(15, training_data, training_refdata)
    print("Network2 err = " + str(network2_err) + ", network3 err = " + str(network3_err))


def find_optimal_layers_neuron_no(training_data, training_refdata):
    network1616 = NeuralNetwork(2, 2)
    network1616.linear_relu_stack = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.ReLU()
    )
    network1616_err = network1616.perform_training(15, training_data, training_refdata)

    network3216 = NeuralNetwork(2, 2)
    network3216.linear_relu_stack = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.ReLU()
    )
    network3216_err = network3216.perform_training(15, training_data, training_refdata)

    network6432 = NeuralNetwork(2, 2)
    network6432.linear_relu_stack = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.ReLU()
    )
    network6432_err = network6432.perform_training(15, training_data, training_refdata)
    print("Network1616 err = " + str(network1616_err) + ", network3216 err = " + str(network3216_err)
          + ", network6416 err = " + str(network6432_err))


def save_to_excel(distribution, file_name):
    df = pd.DataFrame(distribution, columns=['Distribution'])
    df.to_excel('./distributions/' + file_name, index=False)


class NeuralNetwork(nn.Module):

    def __init__(self, inputs_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        vector = self.linear_relu_stack(x)
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
                idx = torch.randperm(training_data.nelement())
                training_data = training_data.view(-1)[idx].view(training_data.size())
                reference_data = reference_data.view(-1)[idx].view(reference_data.size())
            dataset = TensorDataset(training_data, reference_data)
            dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
            for data, refdata in dataloader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, refdata)
                loss.backward()
                optimizer.step()
                return_loss += loss.item()
            print(return_loss/len(dataloader))
        return return_loss

    def predict(self, data):
        output = self(data)
        return output.detach().numpy()

    def find_lr(self, device, trainloader):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=0.000001, weight_decay=5e-5)
        lr_finder = LRFinder(self, optimizer, criterion, device=device)
        lr_finder.range_test(trainloader, end_lr=1, num_iter=500)
        lr_finder.plot()
        lr_finder.reset()


if __name__ == '__main__':
    train_data = DatasetUWB.DatasetUWD(8)
    train_data.import_static_data()
    coords_train, reference_train = train_data.get_torch_dataset()

    test_data = DatasetUWB.DatasetUWD(8)
    test_data.import_file("./dane/pomiary/F8/f8_2p.xlsx")
    coords_test, reference_test = test_data.get_torch_dataset()

    network = NeuralNetwork(2, 2)

    # Dobieranie odpowiedniego współczynnika uczenia
    # dataset = TensorDataset(coords_train, reference_train)
    # dataloader = DataLoader(dataset)
    # network.find_lr("cuda", dataloader)

    # Dobieranie optymalnej liczby warstw ukrytych
    # find_optimal_layers_no(coords_test, reference_test)

    # Dobieranie optymalnej liczby neuronów w warstwach ukrytych
    # find_optimal_layers_neuron_no(coords_test, reference_test)

    network.perform_training(15, coords_train, reference_train)

    for i in range(1, 4):
        test_data.clear()
        test_data.import_file(
            './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_' + str(
                i) + 'p.xlsx')
        test, ref_test = test_data.get_torch_dataset()
        out = network.predict(test)
        dist = Errors.calc_distribution(ref_test, out)
        dist_org = Errors.calc_distribution(ref_test, test)
        save_to_excel(dist, 'distribution_f' + str(test_data.audience_no) + '_' + str(i) + 'p.xlsx')
        Animation.distribution_plot(dist, dist_org,
                                    'distribution_f' + str(test_data.audience_no) + '_' + str(i) + 'p.png')
        Animation.track_plot(test, ref_test, out,
                             'track_f' + str(test_data.audience_no) + '_' + str(i) + 'p.png')

    for i in range(1, 4):
        test_data.clear()
        test_data.import_file(
            './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_' + str(
                i) + 'z.xlsx')
        test, ref_test = test_data.get_torch_dataset()
        out = network.predict(test)
        dist = Errors.calc_distribution(ref_test, out)
        dist_org = Errors.calc_distribution(ref_test, test)
        save_to_excel(dist, 'distribution_f' + str(test_data.audience_no) + '_' + str(i) + 'z.xlsx')
        Animation.distribution_plot(dist, dist_org,
                                    'distribution_f' + str(test_data.audience_no) + '_' + str(i) + 'z.png')
        Animation.track_plot(test, ref_test, out,
                             'track_f' + str(test_data.audience_no) + '_' + str(i) + 'z.png')

    for i in range(1, 3):
        test_data.clear()
        test_data.import_file(
            './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_random_' + str(
                i) + 'p.xlsx')
        test, ref_test = test_data.get_torch_dataset()
        out = network.predict(test)
        dist = Errors.calc_distribution(ref_test, out)
        dist_org = Errors.calc_distribution(ref_test, test)
        save_to_excel(dist, 'distribution_f' + str(test_data.audience_no) + '_random_' + str(i) + 'p.xlsx')
        Animation.distribution_plot(dist, dist_org,
                                    'distribution_f' + str(test_data.audience_no) + '_random_' + str(i) + 'p.png')
        Animation.track_plot(test, ref_test, out,
                             'track_f' + str(test_data.audience_no) + '_random_' + str(i) + 'p.png')
