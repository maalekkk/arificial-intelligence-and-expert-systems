import torch
import pandas as pd
from statistics import mean
from math import fabs
import numpy as np
import matplotlib.pyplot as plt

import Animation
import DatasetUWB
import Errors
from network import Network

# data = pd.read_csv('data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])
# test_data = pd.read_csv('test_data.csv', sep=';', names=['X', 'Y', 'TrueX', 'TrueY'])

train_data = DatasetUWB.DatasetUWD(8)
train_data.import_static_data()
train_data.import_random_data()

test_data = DatasetUWB.DatasetUWD(8)
test_data.import_file(
    './dane/pomiary/F' + str(test_data.audience_no) + '/f' + str(test_data.audience_no) + '_random_' + str(
                1) + 'p.xlsx')

train = torch.tensor(train_data.test, dtype=torch.float)
test = torch.tensor(test_data.test, dtype=torch.float, requires_grad=False)
min = torch.min(train)
max = torch.max(train)
test_min = torch.min(test)
test_max = torch.max(test)

train = (train - min) / (max - min)
test = (test - test_min) / (test_max - test_min)

net = Network()
learning_rate = 2.07E-03
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), learning_rate)

for i in range(100):
    trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.8)
    errors = []
    for x in trainset:
        net.zero_grad()
        input = x[:, [0, 1]]
        target = x[:, [2, 3]]

        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        errors.append(loss.item())
        loss.backward()
        optimizer.step()
    # plt.scatter(i, mean(errors))
    # plt.pause(0.05)
    # if i % 10 == 0:
    print('[', i, ']', 'loss: ', mean(errors))

testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
error_fn = torch.nn.MSELoss()
errors = []
results = []
for x in testset:
    input = x[:, [0, 1]]
    target = x[:, [2, 3]]
    output = net(input)
    input = input * (test_max - test_min) + test_min
    target = target * (test_max - test_min) + test_min
    output = output * (test_max - test_min) + test_min
    row = [
        input[0][0].item(), input[0][1].item(), target[0][0].item(), target[0][1].item(), output[0][0].item(),
        output[0][1].item()
    ]
    results.append(row)

testx = []
testy = []
referencex = []
referencey = []
outx = []
outy = []

for i in results:
    testx.append(i[0])
    testy.append(i[1])
    referencex.append(i[2])
    referencey.append(i[3])
    outx.append(i[4])
    outy.append(i[5])
dist = Errors.calc_distribution(list(zip(referencex, referencey)), list(zip(outx, outy)))
dist_org = Errors.calc_distribution(list(zip(referencex, referencey)), list(zip(testx, testy)))
Animation.distribution_plot(dist, dist_org,
                            'distribution_TEST.png')
Animation.track_plotTEST(testx, testy, referencex, referencey, outx, outy, 'test_track.png')
results = pd.DataFrame(results, columns=['X', 'Y', 'TargetX', 'TargetY', 'OutputX', 'OutputY'])
print(results.head(10))
