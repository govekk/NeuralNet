import csv

# data object takes in a list of info about wine
# stores info and creates getter methods for the features


class WineData:
    def __init__(self, wine_info):
        self.features = wine_info[:-1]
        self.quality = wine_info[-1]

    def get_fixed_acidity(self):
        return self.features[0]

    def get_volatile_acidity(self):
        return self.features[1]

    def get_citric_acid(self):
        return self.features[2]

    def get_residual_sugar(self):
        return self.features[3]

    def get_chlorides(self):
        return self.features[4]

    def get_free_SO2(self):
        return self.features[5]

    def get_total_SO2(self):
        return self.features[6]

    def get_density(self):
        return self.features[7]

    def get_pH(self):
        return self.features[8]

    def get_sulphates(self):
        return self.features[9]

    def get_alcohol(self):
        return self.features[10]

    # this is our output variable
    def get_quality(self):
        return self.quality


# import data from one file and create a list of wine_data objects for it
def import_data_file(file_name):
    data_file = open(file_name, 'r')
    data = csv.reader(data_file, delimiter=';')
    data_list = []
    first_row = True
    for row in data:
        # first row will be header, don't want that data
        if first_row:
            first_row = False
        # for the rest of the rows, create a wine_data object and append to list
        else:
            data_list.append(WineData(row))
    return data_list


# import data for all training, dev, and test files for use later
def import_data():
    training_data = import_data_file("winequality-white.csv")
    # dev_data = import_data_file("dev_data.csv")
    # test_data = import_data_file("test_data.csv")
    print training_data[0].get_chlorides()

if __name__ == "__main__":
    import_data()