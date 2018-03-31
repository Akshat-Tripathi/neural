# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:22:08 2018

@author: akshat
"""

import matrix as m
from math import exp
from random import shuffle

class neural_layer:
    def __init__(self, inputs, activation, nodes, learning_rate):
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
        #Initialises a matrix with random numbers of shape (number of neurons) by (number of input columns)
        self.weights = m.random_matrix([nodes, inputs.shape[0]])
        #Initialises a matrix with random numbers of shape (number of neurons) by (1)
        self.bias = m.random_matrix([nodes, 1])
        #Dictionary containing current activation functions
        activations = {"relu": lambda x : max(0, x),
                       "linear" : lambda x : x,
                       "sigmoid" : lambda x : 1/(1+exp(-x))}
        #Dictionary containing current activation functions' derivatives
        derivs = {"relu": lambda x : 1 if x>0 else 0,
                  "linear" : lambda x : 1,
                  "sigmoid" : lambda x : x*(1-x)}
        #Decides the activation function based off the input from the user
        self.act = activations[activation]
        self.deriv = derivs[activation]
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
        self.output = (self.weights.dot_product(self.inputs) + self.bias).apply_function(self.act)
        return self.output
    
    def stochastic_error(self, ideal):
        """
        Returns the difference between the predicted output and the real output
        """
        return (ideal - self.output)
    
    def update(self, error):
        """
        Used to incrementally update the layer's weights and biases to allow it to perform better
        error is the value returned from the stochastic_error function
        """
        self.error = error
        self.delta = ((self.error*self.output.apply_function(self.deriv))*self.lr)
        self.weights += self.delta.dot_product(self.inputs.Transpose())
        self.bias += self.delta
        return self.delta
    
    def backpropagate(self):
        """Used to pass error backwards through a network"""
        self.back = self.weights.Transpose().dot_product(self.delta)
        return self.back

class neural_network:
    def __init__(self, inputs, targets, layers, error_listener = True, shuffle_enabled = True):
        """
        Layers should be vectors with the variables: activation, nodes, learning rate
        inputs is a 3d array of the input data
        targets is a 3d array of the output data
        layers is a 2d array containing the parameters for the layers - outlined above
        error_listener is a boolean value indicating whether or not the error should be recorded
        shuffle_enabled is a boolean value indicating whether or not the inputs should be randomly ordered
        """
        
        if error_listener:
            self.error_list = []
        self.inputs = inputs
        self.shuffle = shuffle_enabled
        self.targets = targets
        #Initialise network - forward
        self.layers = []
        inp = m.matrix(inputs[0])
        for i in layers:
            act = i[0]
            nodes = i[1]
            lr = i[2]
            layer = neural_layer(inp, act, nodes, lr)
            self.layers.append(layer)
            inp = layer.feed()
        #Initialise network - backward
        out = targets[0]
        e = self.layers[-1].stochastic_error(m.matrix(out))
        for i in self.layers[::-1]:
            i.update(e)
            e = i.backpropagate()
    
    def forward_prop(self, x):
        """
        Returns an output based off the input (x) supplied to it
        Can be used as a predict function
        x is a list containing 1 row of the input
        """
        inp = m.matrix(x)
        for layer in self.layers:
            layer.pass_inputs(inp)
            inp = layer.feed()
        return inp #At this point, inp is the output
    
    def train(self, epochs, batch_size):
        """
        Returns the error list if error_listener is True
        batch_size is an integer defining the number of datapoints used to calculate the error
        Is responsible for training the neural network
        epochs is the number of iterations that the network should train for
        """
        x = self.inputs
        y = self.targets
        self.batch_size = batch_size
        for epoch in range(epochs):
            if self.shuffle:
                c = list(zip(x, y))
                shuffle(c)
                x, y = zip(*c)
                del c
            #initialise error and and final layer out variables
            final = self.layers[-1]
            err = final.feed().apply_function(lambda x:0)
            output = final.feed().apply_function(lambda x:0)
            #Feedforward
            for i in range(self.batch_size):
               inp = x[i]
               output += self.forward_prop(inp)
               err += final.stochastic_error(m.matrix(y[i]))
            #Get error
            ideal = err.apply_function(lambda x: x/self.batch_size)
            if self.error_list != None:
                self.error_list.append(ideal.matrix[0][0])
            
            #Update weights
            for layer in self.layers[::-1]:
                layer.update(ideal)
                ideal = layer.backpropagate()
        try:
            return self.error_list
        except NameError: 
            pass
