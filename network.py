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
import data_setup
from math import exp

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
    def vectorized_sigmoid(self, z_vector):
        a_vector = np.array([0.0 for i in range(len(z_vector))])

        for i in range(len(z_vector)):
            a_vector[i] = self.sigmoid(z_vector[i])

        return a_vector

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
        return (1.0/(1 + exp(-z)))

    """
    Trains the neural network on training data, preparing it for new
    predictions. Uses stochastic gradient descent and backpropagation.

    Parameters:
    :num_epochs: - the number of times to iterate through training data (int)
    :training_data: - the data used to train the network. Comes in a list of
    lists, where each row is a data point and each column is a feature.

    Returns nothing.
    """
    def train(self, training_data, num_epochs = 1000):
        for i in range(num_epochs):
            error2_vectors = []
            error_output_vectors = []
            a1_vectors = []
            a2_vectors=[]
            for training_point in training_data:
                # Initialize first layer of activations (features from training data point)
                a1_vector = np.array([training_point[i] for i in range(len(training_point) - 1)])
                z2_vector, z3_vector, a2_vector = self.feed_forward(a1_vector)
                error_output_vector = self.calculate_output_error(z3_vector, np.array([training_point[-1]]))
                error2_vector = self.backpropagate(error_output_vector, z2_vector)
                error2_vectors.append(error2_vector)
                error_output_vectors.append(error_output_vector)
                a1_vectors.append(a1_vector)
                a2_vectors.append(a2_vector)
            self.gradient_descent(error2_vectors, error_output_vectors, a1_vectors, a2_vectors, len(training_data))


    """
    Iterates through the neural network, calculating the weighted inputs z
    along the way (for a specific training data point).

    Parameters:
    :a1_vector: - activation values for layer 1 (input layer)

    Returns the weighted inputs z in vectors by layer.
    """
    def feed_forward(self, a1_vector):
        # Calculate the weighted inputs z of the hidden layer using the formula:
        #           z^2(vector) = w^2 (matrix) * a^1 (vector) + b2 (vector),
        #            where a^1 = sigmoid(z1)
        z2_vector = np.dot(self.weights[0], a1_vector) + self.biases[0]

        # Calculate a^2 using a^2 = sigmoid(z^2)
        a2_vector = self.vectorized_sigmoid(z2_vector)

        # Calculate z^output using same formula
        z3_vector = np.dot(self.weights[1], a2_vector) + self.biases[1]

        return z2_vector, z3_vector, a2_vector

    """
    Calculates the error of the output neuron (layer 3).

    Compute error vector error_output_vector =
                  cost_function_gradient_vector (*) derivative_output_function(z3_vector)

    Parameters:
    :z3_vector: - the weighted inputs for any neurons in output layer (we expect 1 neuron)
    :label_vector: - the actual known answer of wine quality (int from 0-10)
    """
    def calculate_output_error(self, z3_vector, label_vector):
        return (10*z3_vector - label_vector) * np.array([10])

    """
    Compute the errors vectors for each neuron layer before the output layer
    and after the input layer. For this simple network, there is only one
    hidden layer.

    Compute error2_vector =
                  ((w3 (matrix))^T * error_output_vector)(*) derivative_sigmoid(z2_vector)
        sigmoid'(z) = sigmoid(z)(1 - sigmoid(z))

        note: uses element-wise multiplication

    Parameters:
    :error_output_vector: - vector of errors for the output neurons (we expect 1 neuron).
    :z2_vector: - vector of weighted inputs for the second (hidden) layer.

    Returns error_vector of hidden layer.
    """
    def backpropagate(self, error_output_vector, z2_vector):
        np.transpose(self.weights[1])

        error2_vector = (np.dot(np.transpose(self.weights[1]), error_output_vector)) * \
                        (self.vectorized_sigmoid(z2_vector) * (1 - self.vectorized_sigmoid(z2_vector)))

        return error2_vector

    # performs gradient descent to update weights for one iteration of training
    # input: sum of errors, sum of errors*activations
    # output: none
    # side effects: changes weight and bias instance variables
    def gradient_descent(self, error2_vectors, error_output_vectors, a1_vectors, a2_vectors, num_data_points):
        # layer 2 weights: sum over x: np.dot(error2_vectors[x],np.transpose(a1_vectors[x]))
        #      divide above by num data points, multiply by learning rate, update weights
        # layer 2 biases: sum over x: error2_vectors[x]
        #      divide above by num data points, multiply by learning rate, update weights
        # layer 3 weights: sum over x: np.dot(error_output_vectors[x],np.transpose(a2_vectors[x]))
        #      divide above by num data points, multiply by learning rate, update weights
        # layer 3 biases: sum over x: error_output_vectors[x]
        #      divide above by num data points, multiply by learning rate, update weights

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
    training_data = data_setup.get_training_data()
    myNet = NeuralNetwork()
    myNet.train(training_data, 1)


main()