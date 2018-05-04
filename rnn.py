# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:36:48 2018

@author: akshat
"""

import numpy as np

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
                       "l relu" : lambda x : max(0.01*x, x),
                       "softmax" : lambda x : self.softmax(x)}
        derivs = {"relu": lambda x : 1 if x>0 else 0,
                  "linear" : lambda x : 1,
                  "sigmoid" : lambda x : x*(1-x),
                  "l relu" : lambda x : 1 if x>0 else 0.01,
                  "softmax" : lambda x : self.softmax(x, deriv=True)}

        self.act = np.vectorize(activations[activation])
        self.deriv = np.vectorize(derivs[activation])
        
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
        self.ideal = ideal
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
                
    def softmax(self, x, deriv=False, cross_entropy=False):
        if deriv:
            if cross_entropy:
                return -np.log((np.exp(x) / np.sum(np.exp(x)))[self.ideal, 0]
                