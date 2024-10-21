import numpy as np
import matplotlib.pyplot as plt


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate=0.1, bias_value=1, epochs=10000):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.bias_value = bias_value
        self.epochs = epochs

        # Random initialization of weights
        np.random.seed(42)
        self.weights_input_hidden = np.random.uniform(size=(self.input_neurons, self.hidden_neurons))
        self.weights_hidden_output = np.random.uniform(size=(self.hidden_neurons, self.output_neurons))

        # BIAS only for the output layer, with adjustable value
        self.bias_output = np.ones((1, self.output_neurons)) * self.bias_value

    def forward(self, inputs):
        # Signal flow through the hidden layer (linear activation)
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = self.hidden_layer_input  # No activation function, as it is a linear function

        # Signal flow through the output layer (sigmoid activation)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = sigmoid(self.output_layer_input)

        return self.predicted_output

    def backward(self, inputs, expected_output):
        # Calculate output error
        error = expected_output - self.predicted_output

        # Calculate gradient for the output layer
        d_predicted_output = error * sigmoid_derivative(self.predicted_output)

        # Error for the hidden (linear) layer
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer  # Derivative for linear function is 1

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
        if self.bias_value != 0:
            self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * self.learning_rate

    def train(self, inputs, expected_output):
        errors = []
        for epoch in range(self.epochs):
            # Forward pass
            self.forward(inputs)

            # Backpropagation
            self.backward(inputs, expected_output)

            # Calculate mean error
            error = np.mean(np.abs(expected_output - self.predicted_output))
            errors.append(error)

        return errors
