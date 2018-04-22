# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:22:08 2018

@author: akshat
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:19:19 2018

@author: akshat
"""

import numpy as np
from random import shuffle
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
                       "l relu" : lambda x : max(0.2*x, x)}
        #Dictionary containing current activation functions' derivatives
        derivs = {"relu": lambda x : 1 if x>0 else 0,
                  "linear" : lambda x : 1,
                  "sigmoid" : lambda x : x*(1-x),
                  "l relu" : lambda x : 1 if x>0 else 0.2}

        error_functions = {"discrete": self.discrete_error,
                           "stochastic": self.stochastic_error,
                           "percent": self.percentage_error}
        self.error_function = error_functions[error_function]
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
        self.delta = (self.deriv(self.error*self.output)*self.lr)
        self.weights += np.dot(self.delta, self.inputs.T)
        self.bias += self.delta
        return self.delta

    def backpropagate(self):
        """Used to pass error backwards through a network"""
        self.back = np.dot(self.weights.T, self.delta)
        return self.back
    
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
            f = lambda x:0
            err = f(final.feed())
            output = f(final.feed())
            #Feedforward
            for i in range(self.batch_size):
               inp = x[i]
               output += self.forward_prop(inp)
               err += final.error_function(np.array(y[i]))
            #Get error
            f = lambda x: x/self.batch_size
            ideal = f(err)
            if self.error_list != None:
                self.error_list.append(ideal[0][0])
            if min(self.error_list) == ideal[0][0]:
                self.best = []
                for i in self.inputs:
                    self.best.append(self.forward_prop(i)[0][0])

            #Update weights
            for layer in self.layers[::-1]:
                layer.update(ideal)
                ideal = layer.backpropagate()

        try:
            return self.error_list
        except NameError:
            pass
    
class rnn:
    def __init__(self, nodes, inputs, targets, learning_rate, activation, time_steps, error_listener=True):
        #inputs is a list of (n, 1) numpy arrays as is targets - a list of time steps
        self.nodes = nodes
        self.inputs = inputs
        self.t = time_steps
        self.targets = targets
        self.lr = learning_rate
        
        activations = {"relu": lambda x : max(0, x),
                       "linear" : lambda x : x,
                       "sigmoid" : lambda x : 1/(1+ np.exp(-x)),
                       "l relu" : lambda x : max(0.2*x, x)}
        derivs = {"relu": lambda x : 1 if x>0 else 0,
                  "linear" : lambda x : 1,
                  "sigmoid" : lambda x : x*(1-x),
                  "l relu" : lambda x : 1 if x>0 else 0.2}

        self.act = activations[activation]
        self.deriv = derivs[activation]
        
        if error_listener:
            self.errors = []
        else:
            self.errors = None
        
        #init layers
        self.error = [[] for i in range(self.t)]
        self.hidden = [[] for i in range(self.t)]
        self.outs = [[] for i in range(self.t)]
        
        #init weights and biases
        self.wx = np.random.randn(nodes, len(inputs[0][0]))
        self.wh = np.random.randn(nodes, nodes)
        self.wy = np.random.randn(len(inputs[0][0]), nodes)
        self.bh = np.zeros((nodes, 1))
        self.by = np.zeros((len(inputs[0][0]), 1))
        
    def feed(self, x):
        self.hidden[-1] = np.zeros((self.nodes, 1))
        for i in range(self.t):
            self.hidden[i] = np.tanh(np.dot(self.wx, x[0]) + np.dot(self.wh, self.hidden[i-1]) + self.bh)
            self.outs[i] = self.act(np.dot(self.wy, self.hidden[i]) + self.by)
        return self.outs
    
    def get_error(self, ideal):
        for i in range(len(ideal)):
            self.error[i] = ideal[i] - self.outs[i]
        if self.errors != None:
            self.errors.append(tuple(np.sum(np.abs(i)) for i in self.error))
        return self.error
    
    def backpropagate(self, x):
        #init gradients
        dwx, dwh, dwy = np.zeros_like(self.wx), np.zeros_like(self.wh), np.zeros_like(self.wy)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dnext = np.zeros_like(self.hidden[0])
        
        #backprop
        for i in range(len(self.hidden))[::-1]:
            dwy += np.dot(self.error[i], self.hidden[i].T)
            dby += self.error[i]
            dh = np.dot(self.wy.T, self.error[i]) + dnext
            dbh += self.deriv(self.hidden[i])*dh
            dwx += np.dot(dbh, x[i].T)
            dwh += np.dot(dbh, self.hidden[i-1].T)
            dnext = np.dot(self.wh.T, dbh)
            
        self.wx += dwx * self.lr
        self.wy += dwy * self.lr
        self.wh += dwh * self.lr
        self.bh += dbh * self.lr
        self.by += dby * self.lr
    
    def train(self, epochs):
        for epoch in range(epochs):
            for step in range(len(self.inputs)):
                self.feed(self.inputs[step])
                self.get_error(self.targets[step])
                self.backpropagate(self.inputs[step])
