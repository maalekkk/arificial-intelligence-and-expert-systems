import pandas as pd
import torch
import numpy as np
from constants import COLUMNS


def get_data_from_excel(file_path):
    df = pd.read_excel(file_path, usecols=COLUMNS).values.astype(np.float32)
    ref_data = df[:, 2:]
    data = df[:, :2]
    return data, ref_data


class Dataset:
    def __init__(self, audience_no):
        self.audience_no = audience_no
        self.coords = []
        self.reference = []

    def import_file(self, file_name, err_tolerance=float("inf")):
        data = pd.read_excel(file_name)
        data = pd.DataFrame(data, columns=COLUMNS)
        for i in range(len(data)):
            row = data.iloc[i]
            err = ((row[0] - row[2]) ** 2 + (row[1] - row[3]) ** 2) ** 0.5
            if err < err_tolerance:
                self.coords.append([row[0], row[1]])
                self.reference.append([row[2], row[3]])

    def import_static_data(self, err_tolerance=500):
        for i in range(1, 226):
            file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_stat_" + str(
                i) + ".xlsx"
            self.import_file(file_path, err_tolerance=err_tolerance)
            print('[ ' + str(i) + ' ] : File loaded')
        return torch.from_numpy(np.array(self.coords).astype(np.float32)), torch.from_numpy(
            np.array(self.reference).astype(np.float32))
