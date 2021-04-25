import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import DatasetUWB
import NeuralNetwork
from torch import nn

if __name__ == '__main__':
    dataset = DatasetUWB.DatasetUWD(8)
    dataset.import_data()
    # print(dataset.x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork.NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 2, 2, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
