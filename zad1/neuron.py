import numpy as np
from typing import Callable



def liniar_function(scalar: float, input: float = 1.0) -> float:
    return scalar * input

# budowa neuronu

class Neuron:
    def __init__(self, input: np.ndarray[float] = None, weights: np.ndarray[float] = None,  activation_function: Callable = None, traning_step: float = 0.5, amount_weights_and_neurons = 0.6):
        self.weights = weights
        self.traning_step = traning_step
        self.amount_weights_and_neurons = amount_weights_and_neurons
        if self.weights is None:
            self.weights = np.random.rand(amount_weights_and_neurons)
        self.inputs = input
        if self.inputs is None:
            self.inputs = np.random.rand(amount_weights_and_neurons)
        self.activation_function = activation_function
        if self.activation_function is None:
            self.activation_function = liniar_function

    def get_new_input(self, inputs: np.ndarray[float]):
        self.inputs = inputs
        pass


    def calculate_output(self) -> float:
        return self.activation_function(np.dot(self.inputs, self.weights))
        
    def train_neuron(self, desiaier_output: int):
        new_weights = np.arange(self.amount_weights_and_neurons, dtype=float)
        old_output = self.calculate_output()
        for index, w in enumerate(self.weights):
             new_weights[index] = w + self.traning_step * (desiaier_output - old_output) * self.inputs[index]
        self.weights = new_weights
        pass
