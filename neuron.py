import data_setup
import math


class Neuron:
    def __init__(self):
        self.weights = {}
        self.output = 0
        self.bias = 0

    def setup_weights(self):
        for feature_name in data_setup.headings:
            self.weights[feature_name] = 1

    # takes in dictionary of features
    def input(self, features):
        # sigmoid neuron function
        feature_sum = 0
        for feature_name in features:
            feature_sum += features[feature_name] * self.weights[feature_name]
        self.output = 1/(1+ math.exp(-feature_sum-self.bias))

    def update_weights(self, change):
        # backpropogation
        pass

    def get_output(self, features):
        return self.output