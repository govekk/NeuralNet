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
        b1 = np.array([[randInitialVal()] for j in range(num_hidden)])
        b2 = np.array([[randInitialVal()] for j in range(1)])
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
        a_vector = np.array([[0.0] for i in range(len(z_vector))])

        for i in range(len(z_vector)):
            a_vector[i][0] = self.sigmoid(z_vector[i][0])

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
            if i % 100 == 0:
                print "Training " + str(i) + "th iteration"
            error2_vectors = []
            error3_vectors = []
            a1_vectors = []
            a2_vectors=[]
            for training_point in training_data:
                # Initialize first layer of activations (features from training data point)
                a1_vector = np.array([[training_point[i]] for i in range(len(training_point) - 1)])
                z2_vector, z3_vector, a2_vector = self.feed_forward(a1_vector)
                error_output_vector = self.calculate_output_error(z3_vector, np.array([training_point[-1]]))
                error2_vector = self.backpropagate(error_output_vector, z2_vector)

                error2_vectors.append(error2_vector)
                error3_vectors.append(error_output_vector)
                a1_vectors.append(a1_vector)
                a2_vectors.append(a2_vector)
            self.gradient_descent(error2_vectors, error3_vectors, a1_vectors, a2_vectors, len(training_data))



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
        z2_vector = np.dot(self.weights[0], a1_vector)+ self.biases[0]
        # 6x11 * 11x1

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
        return z3_vector - label_vector

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


    """
    Uses the gradient descent rule to update weights and biases based on one iteration
    through the training data.


    Parameters:
    ::
    """
    def gradient_descent(self, error2_vectors, error3_vectors, a1_vectors, a2_vectors, num_training_data):
        # initialize weight change matrices
        weight2_change_matrix = np.array([[0.0 for k in range(11)] for j in range(self.num_hidden)])
        weight3_change_matrix = np.array([[0.0 for k in range(self.num_hidden)] for j in range(1)])

        # initialize bias change vectors
        bias2_change_vector = np.array([[0.0] for j in range(self.num_hidden)])
        bias3_change_vector = np.array([[0.0] for j in range(1)])

        # sum gradients
        for i in range(num_training_data):
            weight2_change_matrix += np.dot(error2_vectors[i], a1_vectors[i].T)
            weight3_change_matrix += np.dot(error3_vectors[i], a2_vectors[i].T)
            bias2_change_vector += error2_vectors[i]
            bias3_change_vector += error3_vectors[i]

        weight2_change_matrix = weight2_change_matrix * self.learning_rate/num_training_data
        weight3_change_matrix = weight3_change_matrix * self.learning_rate/num_training_data
        bias2_change_vector = bias2_change_vector * self.learning_rate/num_training_data
        bias3_change_vector = bias3_change_vector * self.learning_rate/num_training_data

        self.update_weights([weight2_change_matrix, weight3_change_matrix])
        self.update_biases([bias2_change_vector, bias3_change_vector])
    """
    Updates the weights in accordance with gradient descent rule.

    Parameters:
    :layer: - the layer of weights to update.
    :change_matrix_list: - list of matrices with weight changes (from shallow to deep)

    Returns nothing.
    """
    def update_weights(self, change_matrix_list):
        for i in range(len(change_matrix_list)):
            self.weights[i] -= change_matrix_list[i]


    """
    Updates the biases in accordance with gradient descent rule.

    Parameters:
    :layer: - the layer of weights to update.
    :change_vector: - list of vectors with bias changes (from shallow to deep)

    Returns nothing.
    """
    def update_biases(self, change_vector_list):
        for i in range(len(change_vector_list)):
            self.biases[i] -= change_vector_list[i]

    # input: features of one data point (activation of input) (vector)
    # output: weighted input z of hidden layer (vector),
    #           weighted input z of output layer (vector)


def main():
    training_data = data_setup.get_training_data()
    myNet = NeuralNetwork()
    myNet.train(training_data)


main()