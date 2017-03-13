import numpy

class NeuralNetwork:
    def __init__(self):
        self.weights = [] #0 is hidden, 1 is output
        self.biases = []
        self.output_bias = 0
        self.learning__rate = 0.01

    def init_weights(self):
        pass

    # input: weighted input (z) (vector)
    # output: activation (a) (vector)
    def sigmoid(self, z):
        # iterate in here
        return

    def train(self, num_iter, data_points):
        for i in range(num_iter):
            for data_point in data_points:
                z_hidden, z_output = self.feed_forward(data_point)
                error_output = self.calculate_output_error(z_output, data_point[-1])
                error_hidden = self.backpropagate(error_output, z_hidden)
            self.gradient_descent()

    def update_weights(self, layer, change_vector):
        pass

    def update_biases(self, layer, change_vector):
        pass

    # input: features of one data point (activation of input) (vector)
    # output: weighted input z of hidden layer (vector),
    #           weighted input z of output layer (vector)
    def feed_forward(self, data_point):
        # last column is answer
        # activation of first row are feature values from data_point (vector)
        # calculate z_hidden from data_point, z (vector) = wa (matrix)*(vector) + b (vector)
        # calculate a_hidden from z_hidden in sigmoid function, 1/(1+exp(z))
        #                                    (vector, calculated individually)
        # calculate z_output from a_hidden, z (scaler) = wa (vector)*(vector) + b (scaler)
        z_hidden, a_hidden, z_output = 1
        return z_hidden, z_output

    # input: weighted input z of output neuron (vector length 1), correct output
    # output: error of output layer (vector)
    def calculate_output_error(self, z_output, answer):
        # output derivative = 10 if multiply by 10
        a_output = self.sigmoid(z_output)
        # return (a_output - answer) * 10 < make this a vector!!!

    # backpropagates errors from output to hidden layer
    # input: error of output layer (vector), weighted input of hidden layer (vector)
    # output: error of hidden layer (vector)
    def backpropagate(self, error_output, z_hidden):
        # (self.weights[1] (transposed?) *error_output) * (sig'(z) = (sig(z) (1 - sig(z)))
        pass

    # performs gradient descent to update weights for one iteration of training
    # input: sum of errors, sum of errors*activations
    # output: none
    # side effects: changes weight and bias instance variables
    def gradient_descent(self, ):
        pass