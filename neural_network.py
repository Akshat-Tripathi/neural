# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:36:26 2018

@author: akshat
"""

from neural_net.neural_layer import neural_layer
from random import shuffle
import numpy as np

class neural_network:
    def __init__(self, inputs, targets, layers, error_listener = True, shuffle_enabled = True, error_function = "stochastic"):
        """
        Layers should be vectors with the variables: activation, nodes, learning rate
        inputs is a 3d array of the input data
        targets is a 3d array of the output data
        layers is a 2d array containing the parameters for the layers - outlined above
        error_listener is a boolean value indicating whether or not the error should be recorded
        shuffle_enabled is a boolean value indicating whether or not the inputs should be randomly ordered
        """

        self.error_function = error_function
        if error_listener:
            self.error_list = []
        else:
            self.error_list = None
        self.inputs = inputs
        self.shuffle = shuffle_enabled
        self.targets = targets
        #Initialise network - forward
        self.layers = []
        inp = np.array(inputs[0])
        for i in layers:
            act = i[0]
            nodes = i[1]
            lr = i[2]
            layer = neural_layer(inp, act, nodes, lr, self.error_function)
            self.layers.append(layer)
            inp = layer.feed()
        #Initialise network - backward
        out = targets[0]
        e = self.layers[-1].error_function(np.array(out))
        for i in self.layers[::-1]:
            i.update(e)
            e = i.backpropagate()

    def forward_prop(self, x):
        """
        Returns an output based off the input (x) supplied to it
        Can be used as a predict function
        x is a list containing 1 row of the input
        """
        inp = np.array(x)
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
            #initialise error and final layer out variables
            final = self.layers[-1]
            f = np.vectorize(lambda x:0)
            err = np.float64(f(final.feed()))
            output = np.float64(f(final.feed()))
            #Feedforward
            for i in range(self.batch_size):
               inp = x[i]
               output += self.forward_prop(inp)
               err += final.error_function(np.array(y[i]))
            #Get error
            f = np.vectorize(lambda x: x/self.batch_size)
            ideal = f(err)
            if self.error_list != None:
                self.error_list.append(ideal[0][0])

            #Update weights
            for layer in self.layers[::-1]:
                layer.update(ideal)
                ideal = layer.backpropagate()

        try:
            return self.error_list
        except NameError:
            pass
    