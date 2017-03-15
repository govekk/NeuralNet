"""
data_setup.py
Authors: Ryan Gorey, Kiya Govek

Imports data from training, development, and evaluation csv files
Checks that input data is of the right format (float)
Stores the data in lists
"""

import csv

"""
Changes type of input data to float if it is not already

Parameter: list of feature values for one data point
"""
def check_data_type(wine_info):
    for i in range(len(wine_info)):
        if type(wine_info[i]) != float:
            wine_info[i] = float(wine_info[i])
    return wine_info


"""
import data from one file and create a list of data points from it

Parameter: file name of csv file
"""
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


"""
import data from training, dev, and test files for use later
first method imports from all
later methods import from one file individually

Output: list(s) of data points, where a data point is stored as a list of feature values
"""
def import_data():
    training_data = import_data_file("wines_training.csv")
    dev_data = import_data_file("wines_dev.csv")
    eval_data = import_data_file("wines_eval.csv")
    return training_data, dev_data, eval_data

def get_training_data():
    training_data = import_data_file("wines_training.csv")
    return training_data

def get_dev_data():
    dev_data = import_data_file("wines_dev.csv")
    return dev_data

def get_eval_data():
    eval_data = import_data_file("wines_eval.csv")
    return eval_data


if __name__ == "__main__":
    import_data()