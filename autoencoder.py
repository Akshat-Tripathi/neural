# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:23:30 2018

@author: akshat
"""

from neural_net.neural_network import neural_network
import numpy as np

class autoencoder:
    def __init__(self, inputs, middle, layers, error_listener=True, variational=False):
        self.inputs = inputs
        self.cut = middle
        self.layers = layers
        if variational:
            self.nn = neural_network(self.inputs, variational, self.layers, error_listener=error_listener)
        else:
            self.nn = neural_network(self.inputs, self.inputs, self.layers, error_listener=error_listener)
        y = np.zeros((layers[middle+1][1],1))
        encode_layers = self.layers[:middle+1]
        decode_layers = self.layers[middle+1:]
        self.encoder = neural_network(self.inputs, y, encode_layers, error_listener=False)
        self.decoder = neural_network([y], self.inputs[0], decode_layers, error_listener=False)
    
    def train(self, epochs, batch_size):
        self.nn.train(epochs, batch_size)
    
    def configure(self):
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].weights = self.nn.layers[i].weights
            self.encoder.layers[i].bias = self.nn.layers[i].bias
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i].weights = self.nn.layers[i+self.cut+1].weights
            self.decoder.layers[i].bias = self.nn.layers[i+self.cut+1].bias
        
    def compress(self, data):
        return self.encoder.forward_prop(data)
    
    def decompress(self, compressed_data):
        return self.decoder.forward_prop(compressed_data)