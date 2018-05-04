# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:36:24 2018

@author: akshat
"""

import numpy as np
class neural_layer:
    def __init__(self, inputs, activation, nodes, learning_rate, error_function):
        """
        Initialises a neural layer
        inputs is a row of the input data
        activation is a string which can be either relu, linear or sigmoid
        nodes is the number of neurons in the layer
        """
        #The inputs passed to this layer
        self.inputs = inputs
        #The learning rate of the layer
        self.lr = learning_rate
        #Initialises a matrix with random numbers with a mean of 0 of shape (number of neurons) by (number of input columns)
        self.weights = np.random.randn(nodes, inputs.shape[0])
        #self.weights = m.random_matrix([nodes, inputs.shape[0]])
        #Initialises a matrix with random numbers of shape (number of neurons) by (1)
        self.bias = np.zeros((nodes, 1))
        #Dictionary containing current activation functions
        activations = {"relu": lambda x : max(0, x),
                       "linear" : lambda x : x,
                       "sigmoid" : lambda x : 1/(1+np.exp(-x)),
                       "l relu" : lambda x : max(0.01*x, x)}
        #Dictionary containing current activation functions' derivatives
        derivs = {"relu": lambda x : 1 if x>0 else 0,
                  "linear" : lambda x : 1,
                  "sigmoid" : lambda x : x*(1-x),
                  "l relu" : lambda x : 1 if x>0 else 0.01}

        error_functions = {"discrete": self.discrete_error,
                           "stochastic": self.stochastic_error,
                           "percent": self.percentage_error}
        self.error_function = error_functions[error_function]
        #Decides the activation function based off the input from the user
        self.act = np.vectorize(activations[activation])
        self.deriv = np.vectorize(derivs[activation])
        self.back = None
        self.output = None
        self.error = None
        self.delta = None

    def pass_inputs(self, inputs):
        """
        A neural layer must use this function to handle multiple rows of input data
        inputs is a row of the input data
        """
        self.inputs = inputs

    def feed(self):
        """
        Returns output for the layer, processes the data
        """
        self.output = self.act(np.dot(self.weights, self.inputs) + self.bias)
        return self.output

    def stochastic_error(self, ideal):
        """
        Returns the difference between the predicted output and the real output
        """
        return (ideal - self.output)

    def discrete_error(self, ideal):
        """If successful then 0 else 1"""
        if ideal == self.output:
            return 0
        else:
            return 1

    def percentage_error(self, ideal):
        """Returns the percentage difference"""
        try:
            return (self.output/ideal)-1
        except ZeroDivisionError:
            return -1

    def update(self, error):
        """
        Used to incrementally update the layer's weights and biases to allow it to perform better
        error is the value returned from the stochastic_error function
        """
        self.error = error
        self.delta = (self.error*self.deriv(self.output)*self.lr)
        self.weights += np.dot(self.delta, self.inputs.T)
        self.bias += self.delta
        return self.delta

    def backpropagate(self):
        """Used to pass error backwards through a network"""
        self.back = np.dot(self.weights.T, self.delta)
        return self.back