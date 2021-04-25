import pandas as pd


class DatasetUWD:
    def __init__(self, audience_no):
        self.audience_no = audience_no
        self.coords = []
        self.reference = []

    def import_file(self, file_name):
        data = pd.read_excel(file_name)
        data = pd.DataFrame(data,
                            columns=["data__coordinates__x", "data__coordinates__y", "reference__x", "reference__y"])
        file_coords = []
        file_reference = []
        for i in range(len(data)):
            row = data.iloc[i]
            file_coords.append([row[0], row[1]])
            file_reference.append([row[2], row[3]])
        self.coords.append(file_coords)
        self.reference.append(file_reference)

    def import_static_data(self):
        for i in range(1, 225):
            file_path = "./dane/pomiary/F" + str(self.audience_no) + "/f" + str(self.audience_no) + "_stat_" + str(
                i) + ".xlsx"
            self.import_file(file_path)
