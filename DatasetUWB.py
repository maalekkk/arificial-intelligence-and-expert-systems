import pandas as pd
import torch
import numpy as np


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

    def get_torch_dataset(self):
        return torch.from_numpy(np.array(self.coords).astype(np.float32)), torch.from_numpy(
            np.array(self.reference).astype(np.float32))

    # def get_torch_dataset(self, samples_no):
    #     # for i in range(no_of_last_samples, len(learning_data) + 1):
    #     #     inputs = [data[0] for data in learning_data[i - no_of_last_samples:i]] \
    #     #              + [data[1] for data in learning_data[i - no_of_last_samples:i]]
    #     #     outputs = [learning_data[i - 1][2], learning_data[i - 1][3]]
    #     #     mlp.learn(inputs, outputs, 0.05)
    #     input = []
    #     output = []
    #     for i in range(samples_no, len(self.coords) + 1):
    #         input.append(self.coords[i - samples_no])
    #         output.append(self.reference[i - samples_no])

    def clear(self):
        self.coords = []
        self.reference = []
