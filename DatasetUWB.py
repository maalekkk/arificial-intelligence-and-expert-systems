import pandas as pd
import torch
import numpy as np


class DatasetUWD:
    def __init__(self, audience_no):
        self.audience_no = audience_no
        self.coords = []
        self.reference = []

    def import_file(self, file_name):
        data = pd.read_excel(file_name)
        data = pd.DataFrame(data,
                            columns=["data__coordinates__x", "data__coordinates__y", "reference__x", "reference__y"])

        for i in range(len(data)):
            row = data.iloc[i]
            self.coords.append([row[0], row[1]])
            self.reference.append([row[2], row[3]])

    def import_static_data(self):
        for i in range(1, 25):
            file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_stat_" + str(
                i) + ".xlsx"
            self.import_file(file_path)

    def get_torch_dataset(self):
        return torch.from_numpy(np.array(self.coords).astype(np.float32)), torch.from_numpy(
            np.array(self.reference).astype(np.float32))


if __name__ == '__main__':
    d = DatasetUWD(8)
    d.import_file("./dane/pomiary/F8/f8_stat_1.xlsx")
    c, r = d.get_torch_dataset()
    print(c)
    print(r)
