import numpy as np
from typing import Callable


def linear_function(scalar: float, input: float = 1.0) -> float:
    return scalar * input


class Neuron:
    def __init__(self, inputs: np.ndarray = None, weights: np.ndarray = None,
                 activation_function: Callable = None):
        self.inputs = inputs
        self.weights = weights
        self.activation_function = activation_function if activation_function else linear_function

    def calculate_output(self) -> float:
        return self.activation_function(np.dot(self.inputs, self.weights))

    def predict(self, input_vector: np.ndarray) -> float:
        """ Calculate the activation of the neuron based on a new input vector. """
        self.inputs = input_vector
        return self.calculate_output()

    def __repr__(self):
        return (f"Neuron object, weights: {self.weights}, {self.weights.shape}, inputs: {self.inputs},"
                f" {self.inputs.shape}")
