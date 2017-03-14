"""
Wine Quality - Regression

Authors: Ryan Gorey, Kiya Govek

This program predicts the quality of wine, and then evaluates its
predictions. Uses a neural net to make its predictions.
"""

### PACKAGES ###

import numpy as np
import random
import data_setup
import math
import network

### Main Program ###

def test_hyperparameters():
    training_data = data_setup.get_training_data()
    dev_data = data_setup.get_dev_data()

    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]
    hidden_neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in range(len(learning_rates)):
        for j in range(len(hidden_neurons)):
            wine_net = NeuralNetwork(learning_rates[i], hidden_neurons[j])
            wine_net.train(training_data)
            results = wine_net.get_predictions(dev_data)
            evaluate(results)

def main():

    # Get Data
    training_data = data_setup.get_training_data()

    # Make Neural Network Object
    wine_net = network.NeuralNetwork()
    wine_net.train(training_data, 10000)
    # Evaluate Dev Data Options
    dev_data = data_setup.get_dev_data()
    preds = wine_net.get_predictions(dev_data)

    MSE = 0



    for i in range(len(preds)):
        MSE += (float(preds[i][0]) - float(preds[i][1])) ** 2
    print(math.sqrt(MSE / len(preds)))

    # Choose Favorite

    # Evaluate Eval Data


main()