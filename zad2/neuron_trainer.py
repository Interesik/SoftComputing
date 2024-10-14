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
                self.train_neuron_with_set(neuron)
        pass

    def show_calculate_results(self):
        for training_index, training_input in enumerate(self.test_set_inputs):
            self.neurons[0].get_new_input(training_input)
            print("result of 1 neuron for training {} : {}, true class {}".format(training_index, self.neurons[0].calculate_output(), self.test_set_outputs[training_index]))
        pass
    

    def show_weights(self):
        for index, neuron in enumerate(self.neurons):
            print("Weights of {} neuron : {}".format(index, neuron.weights))
        pass

    def train_neuron_with_set(self, neuron: Neuron):
        for training_index, training_vector in enumerate(self.training_inputs[0]):
            neuron.get_new_input(inputs = training_vector)
            neuron.train_neuron(self.training_outputs[training_index])
        pass
    
    def set_other_neurons(self, neurons: list[Neuron]):
        self.neurons = neurons
        pass

    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return round(dot_product / (norm_a * norm_b),3)

