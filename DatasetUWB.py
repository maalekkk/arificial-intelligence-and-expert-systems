import pandas as pd


class DatasetUWD:

    def __init__(self, audience_no):
        self.audience_no = audience_no
        self.x = []
        self.y = []
        self.z = []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.y[index], self.z[index]]

    def import_file(self, file_no):
        file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_stat_" + str(
            file_no) + ".xlsx"
        data = pd.read_excel(file_path)
        data = pd.DataFrame(data, columns=["data__coordinates__x", "data__coordinates__y", "data__coordinates__z"])
        self.x.append(data["data__coordinates__x"].tolist())
        self.y.append(data["data__coordinates__y"].tolist())
        self.z.append(data["data__coordinates__z"].tolist())

    def import_data(self):
        for i in range(1, 225):
            self.import_file(i)
