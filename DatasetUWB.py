import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DatasetUWD:
    def __init__(self, audience_no):
        self.audience_no = audience_no
        self.coords = []
        self.reference = []

    def import_file(self, file_name, err_tolerance=float("inf")):
        data = pd.read_excel(file_name)
        data = pd.DataFrame(data,
                            columns=["data__coordinates__x", "data__coordinates__y", "reference__x", "reference__y"])

        for i in range(len(data)):
            row = data.iloc[i]
            err = ((row[0] - row[2]) ** 2 + (row[1] - row[3]) ** 2) ** 0.5
            if err < err_tolerance:
                self.coords.append([row[0], row[1]])
                self.reference.append([row[2], row[3]])
            else:
                # print("Debug")
                pass

    def import_static_data(self, err_tolerance=500):
        for i in range(1, 226):
            file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_stat_" + str(
                i) + ".xlsx"
            self.import_file(file_path, err_tolerance=err_tolerance)

    def import_random_data(self, err_tolerance=500):
        for i in range(1, 3):
            file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_random_" + str(
                i) + "p.xlsx"
            self.import_file(file_path, err_tolerance=err_tolerance)

    def get_torch_dataset(self):
        return torch.from_numpy(np.array(self.coords).astype(np.float32)), torch.from_numpy(
            np.array(self.reference).astype(np.float32))

    def get_data_loader(self):
        coords = torch.tensor(self.coords, dtype=torch.float)
        reference = torch.tensor(self.reference, dtype=torch.float)
        dataset = TensorDataset(coords, reference)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
        return dataloader

    def clear(self):
        self.coords = []
        self.reference = []
