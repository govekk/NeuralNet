import csv

# # data object takes in a list of info about wine
# # stores info in a dictionary
# # getter method for the features takes in a feature name - must be same as in headings
# class WineDataPoint:
#     def __init__(self, wine_headings, wine_info):
#         self.features = self.set_features(wine_headings[:-1], wine_info[:-1])
#         self.quality = float(wine_info[-1])
#
#     def set_features(self, wine_headings, wine_info):
#         self.check_inputs(wine_headings, wine_info)
#         features = {}
#         for i in range(len(wine_headings)):
#             features[wine_headings[i]] = wine_info[i]
#         return features
#
#     def check_inputs(self, wine_headings, wine_info):
#         for i in range(len(wine_headings)):
#             if type(wine_info[i]) != float:
#                 wine_info[i] = float(wine_info[i])
#             if type(wine_headings) != str:
#                 wine_headings[i] = float(wine_headings[i])
#
#     def get_feature(self, feature_name):
#         if feature_name in self.features:
#             return self.features[feature_name]
#         else:
#             return None
#
#     # this is our output variable
#     def get_quality(self):
#         return self.quality


# each feature is a float, so change from string to float
def check_data_type(wine_info):
    for i in range(len(wine_info)):
        if type(wine_info[i]) != float:
            wine_info[i] = float(wine_info[i])
    return wine_info


# import data from one file and create a list of data points from it
def import_data_file(file_name):
    data_file = open(file_name, 'r')
    data = csv.reader(data_file, delimiter=',')
    data_list = []
    first_row = []
    for row in data:
        # first row will be header, don't want that data
        if not first_row:
            first_row = row
        # for the rest of the rows, create a data point object and append to list
        else:
            data_list.append(check_data_type(row))
    return data_list


# import data for all training, dev, and test files for use later
def import_data():
    training_data = import_data_file("wines_training.csv")
    dev_data = import_data_file("wines_dev.csv")
    test_data = import_data_file("wines_eval.csv")
    return training_data, dev_data, test_data

if __name__ == "__main__":
    import_data()