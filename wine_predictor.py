"""
Wine Quality - Regression

Authors: Ryan Gorey, Kiya Govek

This program predicts the quality of wine, and then evaluates its
predictions. Uses a neural net to make its predictions.
"""

### PACKAGES ###

import data_setup
import math
import network

### Main Program ###
"""
Calculates the RMSE of the wine predictor.

Parameters:
:preds: - a list of lists, where each sublist contains two values. The first value is the
predicted quality, and the second value is the actual wine quality.

Returns the RMSE for the predictions.
"""
def getRMSE(preds):
    MSE = 0
    for x in range(len(preds)):
        MSE += (float(preds[x][0]) - float(preds[x][1])) ** 2
    RMSE = math.sqrt(MSE / len(preds))

    return RMSE

def main():
    wine_net = network.NeuralNetwork(learning_rates=0.1, hidden_neurons=10)
    wine_net.train(training_data, batch_size=256, num_epochs=1000)
    preds = wine_net.get_predictions(dev_data)
    print(getRMSE(preds))

main()