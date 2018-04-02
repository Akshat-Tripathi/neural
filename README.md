# Neural Network

This is a template of a fully customisable neural network, written fully in Python. It allows anyone with limited knowledge of machine learning to easily create a working neural network. While allowing those with a more sound knowledge of the field to create more complex networks.

# Requirements:
This module only requires matrix, which can be found [here](https://github.com/Akshat-Tripathi/matrix-lib)

# Installation:
Clone this repository with the following command.

``` https://github.com/Akshat-Tripathi/neural_net.git```

# Learning algorithms:
Currently neural networks can be made using:
* Backpropagation

# Objects

## Neural layer

### Inputs:
* inputs: 1 row of the dataset in the form of the matrix data type.
* activation: The activation function of the layer can be either:
    1. relu
    2. sigmoid
    3. linear
    
These must be lowercase strings
* nodes: The number of neurons in the layer. Must be integers.
* learning_rate: The learning rate of the layer, must be an integer or a float.

```Python 
pass_inputs(self, inputs)
```
This method used to allow the layer to accept another row of input data.

The layer is initialised with only a single row of data for example: [1, 2, 3], and to pass a new row [4, 5, 6], this method must be used.

inputs: 1 row of the dataset in the form of a matrix data type.

```Python 
feed(self)
```
This method processes the input data and returns an output.

Note: the output is already an instance of the matrix data type. 

```Python 
stochastic_error(self, ideal)
```
This method returns the difference between the output of the layer and the ideal value. of the output.

ideal: The ideal value of the output of the layer in the form of the matrix data type.

```Python 
update(self, error)
```
This method incrementally updates the weight and bias matrices of the layer.

error: The value returned from the stochastic_error method

Returns the matrix used to update the weights and biases

```Python 
backpropagate(self)
```
This method is used to generate the error for the previous layer.

## Neural network

### Inputs:
inputs: A 3d array of the input data

targets: A 3d array of the output data

layers: A 2d array containing the parameters for the layers - should be vectors with the variables: [activation, nodes, learning rate]

error_listener: A boolean value indicating whether or not the error should be recorded

shuffle_enabled: A boolean value indicating whether or not the inputs should be randomly ordered

```Python 
forward_prop(self, x)
```
This method propagates the input variable x through the network and returns the prediction of the network.

This method should be used to output the prediction of the network.

x: A single row of the input data in the form of the matrix data type

```Python 
train(self, epoch, batch_size)
```
This method trains the neural network on all of the input data.

batch_size: An integer defining the number of datapoints used to calculate the error

epoch: The number of times the network should iterate over the dataset

Returns a list of the error over the epochs if error_listener is enabled

# Usage guide
This first part will outline the creation of a neural network by using only the neural layers object.

First we want to import the necessary modules and create the input and target data.
```Python
from neural_net import neural_layer
import matrix as m
x = [[[0], [1]],
     [[1], [0]],
     [[1], [1]],
     [[0], [0]]]

y = [[[1]],
     [[1]],
     [[0]],
     [[0]]]
```

Then the network is intialised with the creation of the layers. The network will be a two layer neural network with the relu activation function and a learning rate of 0.1.
```Python
nn1 = neural_layer(m.matrix(x[0]), "relu", 5, 0.1)
nn2 = neural_layer(nn1.feed(), "relu", 1, 0.1)
```
Notice how the number of neurons in `nn2` is equal to the number of outputs of the network. Tis must always be done for the network to function.

Now for the network to be fully initialised, the backwards dataflow must be created.
```Python
error = nn2.stochastic_error(m.matrix(y[0]))
nn2.update(error)
error = nn2.backpropagate()
nn1.update(error)
```

Now the network has been initialised, it must be trained. Training involves the forward and backward propagation of the data. 

These have already been done to initialise the network, and are to be repeated.
```Python
for epoch in range(100):
    inp = x[i%4]
    nn1.pass_inputs(m.matrix(inp))
    nn2.pass_inputs(nn1.feed())
    error = nn2.stochastic_error(m.matrix(y[0]))
    nn2.update(error)
    error = nn2.backpropagate()
    nn1.update(error)
```
This completes the training process and produces a basic neural network.

## This second part outlines the creation of the neural network using the neural_network object.

First we again want to import the necessary modules and create the input and target data.
```Python
from neural_net import neural_network
x = [[[0], [1]],
     [[1], [0]],
     [[1], [1]],
     [[0], [0]]]

y = [[[1]],
     [[1]],
     [[0]],
     [[0]]]
```

Then we create a list to store the information about the layers.
```Python
layers = [["relu", 5, 0.1],
          ["relu", 1, 0.1]]
```
The network above has the same shape as the previous network.

Now we create the neural_network and train it.
``` Python
nn = neural_network(x, y, layers, error_listener=True)
error = nn.train(1000, 3)
```
The error variable in this case is a list of the error per epoch the network was trained.
