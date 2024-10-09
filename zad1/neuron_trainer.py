# class to learn neurons it's weights
from neuron import *



class NeuronTrainer:
    def __init__(self, training_inputs: np.ndarray[float], training_outputs: np.ndarray[int], neurons: list[Neuron] = None, test_set_inputs = np.ndarray[float] , test_set_outputs = np.ndarray[int], steps: int = 1000 ):
        self.neurons = neurons
        if self.neurons is None:
            self.neurons = [Neuron(input = training_inputs[0])]
        self.training_inputs = training_inputs,
        self.training_outputs = training_outputs
        if(training_inputs.shape[0] != training_outputs.size):
            raise Exception("Number of traning vectors is not same as desirable outputs")
        self.test_set_inputs = test_set_inputs,
        self.test_set_outputs = test_set_outputs,
        self.steps = steps

    def train_neurons(self):
        for _ in range(self.steps):
            for neuron in self.neurons:
                self.train_neruon_with_set(neuron)
        pass

    def show_calculate_results(self):
        for training_index, training_input in enumerate(self.test_set_inputs):
            self.neurons[0].get_new_input(training_input)
            print("result of 1 neuron for traning {} : {}, true class {}".format(training_index, self.neurons[0].calculate_output(), self.test_set_outputs[training_index]))
        pass
    

    def show_weights(self):
        for index, neuron in enumerate(self.neurons):
            print("Weights of {} neuron : {}".format(index, neuron.weights))
        pass

    def train_neruon_with_set(self, neuron: Neuron):
        for traning_index,traning_vector in enumerate(self.training_inputs[0]):
            neuron.get_new_input(inputs = traning_vector)
            neuron.train_neuron(self.training_outputs[traning_index])
        pass
    
    def set_other_neurons(self, neurons: list[Neuron]):
        self.neurons = neurons
        pass
