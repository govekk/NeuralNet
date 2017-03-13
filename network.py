"""
Neural Network Implementation - Wine Quality Regression Task

Authors: Ryan Gorey, Kiya Govek

This program implements a basic neural net, harcoded to use one layer
of hidden neurons. This implementation uses gradient descent and
backpropagation to learn from training data, and then can predict
the quality of new wines.
"""

### PACKAGES ###

import numpy as np
import random

### HELPER FUNCTIONS ###

"""
Generates a random real number between -1 and 1.

Parameters: None

Returns a real number between -1 and 1.
"""
def randInitialVal():
    return random.uniform(-1, 1)

### NEURAL NETWORK ###

"""
Neural Network Class:

Implements a neural network that uses gradient descent and
backpropagation to learn from training data.
"""
class NeuralNetwork:

    """
    Constructor: Initializes weights and biases for all neurons as zeros.

    Parameters:
    :learning_rate: learning rate between 0 and 1, initialized at 0.01

    Returns a Neural Network object.
    """
    def __init__(self, learning_rate = 0.01, num_hidden = 6):
        # Initialize weight matrices

        w1 = np.array([[randInitialVal() for k in range(11)] for j in range(num_hidden)])
        w2 = np.array([[randInitialVal() for k in range(num_hidden)] for j in range(1)])
        self.weights = [w1, w2]

        # Initialize bias matrices
        b1 = np.array([randInitialVal() for j in range(num_hidden)])
        b2 = np.array([randInitialVal() for j in range(1)])
        self.biases = [b1, b2]

        self.output_bias = 0
        self.learning_rate = learning_rate

        assert(int(num_hidden) > 0)
        self.num_hidden = int(num_hidden)

    """
    Applies the sigmoid function to vector of weighted inputs z.

    Parameters: Vector of weighted inputs z (floats)

    Returns vector of activations.
    """
    def vectorizedSigmoid(self, z_vector):
        for i in range(len(z_vector)):
            z_vector[i] = sigmoid(z_vector[i])

        return z_vector

    """
    Applies the sigmoid function to a single weighted input z.

    Parameters:
    :z: - the weighted input for a sigmoid neuron

    Returns the activation (float) using sigmoid function:

            1
        _________

        1 + e^(-z)
    """
    def sigmoid(self, z):
        return (1/(1 + exp(-z)))

    """
    Trains the neural network on training data, preparing it for new
    predictions. Uses stochastic gradient descent and backpropagation.

    Parameters:
    :num_epochs: - the number of times to iterate through training data (int)
    :training_data: - the data used to train the network. Comes in a list of
    lists, where each row is a data point and each column is a feature.

    Returns nothing.
    """
    def train(self, num_epochs, training_data):
        for i in range(num_epochs):
            for training_point in training_data:
                z_hidden, z_output = self.feed_forward(training_point)
                error_output = self.calculate_output_error(z_output, training_point[-1])
                error_hidden = self.backpropagate(error_output, z_hidden)
            self.gradient_descent()


    """
    Iterates through the neural network, calculating the weighted inputs z
    along the way (for a specific training data point).

    Parameters:
    :training_point: - a single case of training data

    Returns the weighted inputs z in vectors by layer.
    """
    def feed_forward(self, training_point):
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

    """
    Updates the weights in accordance with gradient descent rule.

    Parameters:
    :layer: - the layer of weights to update.
    :change_vector: - the calculated step change

    Returns nothing.
    """
    def update_weights(self, layer, change_vector):
        pass

    """
    Updates the biases in accordance with gradient descent rule.

    Parameters:
    :layer: - the layer of weights to update.
    :change_vector: - the calculated step change

    Returns nothing.
    """
    def update_biases(self, layer, change_vector):
        pass

    # input: features of one data point (activation of input) (vector)
    # output: weighted input z of hidden layer (vector),
    #           weighted input z of output layer (vector)


def main():
    myNet = NeuralNetwork()
    myNet.train()

main()