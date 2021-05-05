import torch
from torch import nn, optim
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import Errors
import random
import Animation
import DatasetUWB


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

    def perform_training(self, epochs, dataloader, lr=2.07E-03):
        assert (epochs > 0)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=5e-5)

        for epoch in range(epochs):
            running_loss = 0
            for data, ref_data in dataloader:
                idx = torch.randperm(data.nelement())
                training_data = data.view(-1)[idx].view(data.size())
                reference_data = ref_data.view(-1)[idx].view(ref_data.size())
                optimizer.zero_grad()
                output = self(training_data)
                loss = criterion(output, reference_data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(running_loss / len(dataloader))

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
    train = train_data.get_data_loader()

    test_data = DatasetUWB.DatasetUWD(8)
    test_data.import_file("./dane/pomiary/F8/f8_2p.xlsx")
    test, ref_test = test_data.get_torch_dataset()

    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    network = NeuralNetwork(2, 2).to(device)

    # network.find_lr("cuda", train)

    network.perform_training(10, train)

    for i in range(1, 4):
        test_data.clear()
        test_data.import_file(
            './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_' + str(
                i) + 'p.xlsx')
        test, ref_test = test_data.get_torch_dataset()
        out = network.predict(test)
        dist = Errors.calc_distribution(ref_test, out)
        dist_org = Errors.calc_distribution(ref_test, test)
        Animation.distribution_plot(dist, dist_org,
                                    'distribution_f' + str(test_data.audience_no) + '_' + str(i) + 'p.png')
        Animation.track_plot(test, ref_test, out,
                             'track_f' + str(test_data.audience_no) + '_' + str(i) + 'p.png')

    for i in range(1, 3):
        test_data.clear()
        test_data.import_file(
            './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_random_' + str(
                i) + 'p.xlsx')
        test, ref_test = test_data.get_torch_dataset()
        out = network.predict(test)
        dist = Errors.calc_distribution(ref_test, out)
        dist_org = Errors.calc_distribution(ref_test, test)
        Animation.distribution_plot(dist, dist_org,
                                    'distribution_f' + str(test_data.audience_no) + '_random_' + str(i) + 'p.png')
        Animation.track_plot(test, ref_test, out,
                             'track_f' + str(test_data.audience_no) + '_random_' + str(i) + 'p.png')
