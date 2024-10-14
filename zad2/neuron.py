import numpy as np
from typing import Callable

def linear_function(scalar: float, input: float = 1.0) -> float:
    return scalar * input

class Neuron:
    def __init__(self, input: np.ndarray = None, weights: np.ndarray = None,
                 activation_function: Callable = None, training_step: float = 0.1,
                 amount_weights_and_neurons=6):
        self.weights = weights if weights is not None else np.random.uniform(-1, 1, amount_weights_and_neurons)
        self.training_step = training_step
        self.amount_weights_and_neurons = amount_weights_and_neurons
        self.inputs = input if input is not None else np.random.rand(amount_weights_and_neurons)
        self.activation_function = activation_function if activation_function is not None else linear_function

    def train_neuron(self, desired_output: int):
        old_output = self.calculate_output()
        for index in range(self.amount_weights_and_neurons):
            self.weights[index] += self.training_step * (desired_output - old_output) * self.inputs[index]
            # Clip weights to ensure they remain in a reasonable range
            self.weights[index] = np.clip(self.weights[index], -1.0, 1.0)

    def calculate_output(self) -> float:
        return self.activation_function(np.dot(self.inputs, self.weights))

    def predict(self, input_vector: np.ndarray) -> float:
        """ Calculate the activation of the neuron based on a new input vector. """
        self.inputs = input_vector
        return self.calculate_output()
