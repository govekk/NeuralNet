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

    learning_rates = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    hidden_neurons = [2, 4, 6, 8, 10]
    batch_sizes = [128, 256]
    results = []

    counter = 0
    for i in range(len(learning_rates)):
        for j in range(len(hidden_neurons)):
            for k in range(len(batch_sizes)):
                counter += 1
                print("Training Combo: " + str(counter))
                wine_net = None
                wine_net = network.NeuralNetwork(learning_rates[i], hidden_neurons[j])
                wine_net.train(training_data, batch_size=batch_sizes[k], num_epochs=2000)
                preds = wine_net.get_predictions(dev_data)

                MSE = 0
                for x in range(len(preds)):
                    MSE += (float(preds[x][0]) - float(preds[x][1])) ** 2
                RMSE = math.sqrt(MSE) / len(preds)
                results.append([learning_rates[i], hidden_neurons[j], batch_sizes[k], RMSE])

    return results

def save_all_results(results):
    f = open("Results.txt", "w")

    for combo in results:
        f.write("Learning Rate: " + str(combo[0]) + ", Hidden Neurons: " + str(combo[1]) + ", Batch Sizes: " + str(combo[2]) + "\n")
        f.write("RMSE: " + str(combo[3]) + "\n\n")

    f.close()

def save_best_result(results):
    best_RMSE = 999999
    best_RMSE_index = -1
    for i in range(results):
        if results[i][3] < best_RMSE:
            best_RMSE = results[i][3]
            best_RMSE_index = i

    f = open("Best_Result.txt", "w")
    f.write("Learning Rate: " + str(results[i][0]) + ", Hidden Neurons: " + str(results[i][1]) + ", Batch Sizes: " + str(results[i][2]) + "\n")
    f.write("RMSE: " + str(results[i][3]) + "\n\n")
    f.close()

def main():
    results = test_hyperparameters()
    save_all_results(results)
    save_best_result()


main()