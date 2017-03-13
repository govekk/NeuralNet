import numpy

class NeuralNetwork:
    def __init__(self):
        self.weights = [] #0 is hidden, 1 is output
        self.biases = []
        self.output_bias = 0
        self.learning__rate = 0.01

    def init_weights(self):
        pass

    def train(self, num_iter, data_points):
        for i in range(num_iter):
            for data_point in data_points:
                z_hidden, z_output = self.feed_forward(data_point)
                error_output = self.calculate_output_error(z_output, data_point[-1])
                error_hidden = self.backpropagate(error_output, z_hidden, data_point[:-1])
            self.gradient_descent()

    def update_weights(self, layer, change_vector):
        pass

    def update_biases(self, layer, change_vector):
        pass

    def feed_forward(self, data_point):
        # last column is answer
        # activation of first row are feature values from data_point (vector)
        # calculate z_hidden from data_point, z (vector) = wa (matrix)*(vector) + b (vector)
        # calculate a_hidden from z_hidden in sigmoid function, 1/(1+exp(z))
        #                                    (vector, calculated individually)
        # calculate z_output from a_hidden, z (scaler) = wa (vector)*(vector) + b (scaler)
        z_hidden, a_hidden, z_output = 1
        return z_hidden, z_output

    def calculate_output_error(self, a_output, answer):
        # output derivative = 10 if multiply by 10
        return (a_output - answer) * 10

    def backpropagate(self, error_output, z_hidden, a_input):
        # (self.weights[1] (transposed?) *error_output) * sig'(z) = (sig(z) (1 - sig(z))
        pass

    def gradient_descent(self, ):
        pass