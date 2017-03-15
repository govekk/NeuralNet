CS321 Artificial Intelligence final project - Ryan Gorey and Kiya Govek

Neural network to perform regression on wine data uses:
- one hidden layer
- sigmoid neurons
- stochastic gradient descent
- 70% training data, 15% development data, 15% test data

TO RUN:
You will need: Python (3 preferred, but should work on either), Numpy - numpy can be installed easily using pip
In terminal, type:
python wine_predictor.py
(in other words run wine_predictor.py using python with no parameters)

If you would like to look at what hyperparameters we are using or change them, look in the main method of
wine_predictor.py.

This should print to the terminal for every few iterations of training that are run, then print the root mean squared
error of the evaluation data

NOTES:
- To download and divide data yourself, run wine_data_separator.Rmd. 