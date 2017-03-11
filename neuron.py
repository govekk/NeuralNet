import data_setup

class Neuron:
    def __init__(self):
        self.weights = {}
        self.output = 0

    def setup_weights(self):
        for feature_name in data_setup.headings:
            self.weights[feature_name] = 1

    def input(self, feature, feature_name):
        # probably not exactly like this, but this is how you use feature and weights
        self.output += feature * self.weights[feature_name]
        pass

    def update_weights(self):
        # backpropogation
        pass

    def get_output(self, features):
        return self.output