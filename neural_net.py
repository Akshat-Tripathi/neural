# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:22:08 2018

@author: akshat
"""

import matrix as m
from math import exp
class neural_layer:
    def __init__(self, inputs, activation, nodes, learning_rate):
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
        self.inputs = inputs
        
    def feed(self):
        self.output = (self.weights.dot_product(self.inputs) + self.bias).apply_function(self.act)
        return self.output
    
    def stochastic_error(self, ideal):
        self.error = (ideal - self.output)
    
    def update(self, ideal):
        self.error = (ideal-self.output)
        #self.error = self.error.apply_function(abs)
        self.delta = ((self.error*self.output.apply_function(self.deriv))*self.lr)
        self.weights -= self.delta.dot_product(self.inputs.Transpose())
        self.bias -= self.delta
        return self.delta
    
    def backpropagate(self):
        self.back = self.weights.Transpose().dot_product(self.delta)
        return self.back
    
