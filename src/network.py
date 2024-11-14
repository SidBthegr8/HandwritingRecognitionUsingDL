"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_fast(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch_fast(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch."""
        # Initialize gradients for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Prepare mini_batch as an array of inputs and outputs
        inputs = np.squeeze(np.array([x for x, y in mini_batch]))  # shape (m, input_size)
        outputs = np.array([y for x, y in mini_batch])  # shape (m, output_size)
        
        # Feedforward over all examples in the mini-batch
        activations, zs = self.feedforward_batch(inputs)
        # Compute the error for the output layer
        delta = self.cost_derivative(activations[-1], outputs) * sigmoid_prime(zs[-1])
        
        # Compute nabla_b and nabla_w for the output layer
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True)  # Sum over all examples (m, 1)
        nabla_w[-1] = np.dot(delta.T, activations[-2])  # (m, output_size) . (m, hidden_size) -> (output_size, hidden_size)
        print("Delta shape: ", [d.shape for d in delta])
        print("Activations shape: ", [a.shape for a in activations])
        print("Nabla_w shapes: ", [nw.shape for nw in nabla_w])
        # Backpropagate for the hidden layers
        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l+1]) * sigmoid_prime(zs[-l])
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True)  # (hidden_size, 1)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1])  # (m, hidden_size) . (m, input_size) -> (hidden_size, input_size)
        
        # Update the weights and biases
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop_fast(self, inputs, outputs):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward over all examples in the mini-batch
        activations, zs = self.feedforward_batch(inputs)
        
        # Compute error for the output layer
        delta = self.cost_derivative(activations[-1], outputs) * sigmoid_prime(zs[-1])
        
        # Compute nabla_b and nabla_w for the output layer
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True)
        nabla_w[-1] = np.dot(delta.T, activations[-2])
        
        # Backpropagate for the hidden layers
        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l+1]) * sigmoid_prime(zs[-l])
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1])
        
        return (nabla_b, nabla_w)

    def feedforward_batch(self, inputs):
        """Perform feedforward computation for the entire mini-batch at once."""
        activations = [inputs]  # The first layer activations are the inputs themselves
        zs = []  # List to store the z vectors for each layer
        
        # Iterate through each layer (except the input layer)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(inputs, w.T) + b.T  # Matrix multiplication + bias addition
            zs.append(z)
            inputs = sigmoid(z)  # Apply the activation function (sigmoid in this case)
            activations.append(inputs)  # Store the activations for each layer
        
        return activations, zs

    def cost_derivative_batch(self, output_activations, y):
        output_activations = [np.reshape(arr, (10, 1)) for arr in output_activations]
        print("Shape of Output_activations: ", [oac.shape for oac in output_activations])
        print(output_activations[0])
        print("Shape of y: ", [o.shape for o in y])
        
        print("Shape of outputs: ", [o.shape for o in output_activations-y])
        return (output_activations-y)  
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)              

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))