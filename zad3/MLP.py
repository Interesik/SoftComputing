import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses):
    plt.plot(losses)
    plt.title('Loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000, bias=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output = None
        self.output_input = None
        self.hidden_output = None
        self.hidden_input = None
        self.input = None
        # Initialize random weights
        self.w_hidden = np.random.randn(input_size, hidden_size)
        self.w_output = np.random.randn(hidden_size, output_size)
        # self.bias_hidden = np.random.randn(hidden_size)
        # self.bias_output = np.random.randn(output_size)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias

        # history of weights for plots
        self.weight_history_hidden = []
        self.weight_history_output = []
        self.accuracies = []

        self.add_bias()

    def forward(self, x):
        self.input = x
        # Hidden layer
        self.hidden_input = np.dot(self.input, self.w_hidden)
        self.hidden_output = np.dot(self.hidden_input, self.w_output[0:2])
        # Output layer
        self.output_input = np.dot(self.hidden_output, self.w_output)
        self.output = sigmoid(self.output_input)
        return self.output

    def add_bias(self):
        self.w_output = np.append(self.w_output, [self.bias for _ in range(self.output_size)])
        self.w_output = self.w_output.reshape(3, 4)

    def backward(self, y_true):
        # Calculating error from output layer
        output_error = y_true - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculating error from hidden output
        hidden_error = np.dot(output_delta, self.w_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Weights update
        self.w_output += self.learning_rate * np.dot(self.hidden_output.reshape(-1, 1), output_delta.reshape(1, -1))
        self.bias_output += self.learning_rate * output_delta
        self.w_hidden += self.learning_rate * np.dot(self.input.reshape(-1, 1), hidden_delta.reshape(1, -1))
        self.bias_hidden += self.learning_rate * hidden_delta

        return output_error

    def train(self, X, y):
        losses = []
        for epoch in range(self.epochs):
            loss = 0
            correct_predictions = 0
            for i in range(len(X)):
                # Forward pass
                y_pred = self.forward(X[i])

                # Backward pass
                output_error = self.backward(y[i])
                loss += np.mean(np.square(output_error))

                # Checking accuracy of prediction
                if np.array_equal(np.round(y_pred), y[i]):
                    correct_predictions += 1

            # calculating overall accuracy
            accuracy = correct_predictions / len(X)
            self.accuracies.append(accuracy)

            # Adding historical weights
            self.weight_history_hidden.append(np.copy(self.w_hidden))
            self.weight_history_output.append(np.copy(self.w_output))

            # Loss function
            losses.append(loss / len(X))
            if self.epochs % 1000 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss / len(X)}, Accuracy: {accuracy}')

        # Plot drawing
        plot_loss(losses)
        self.plot_accuracy()
        self.plot_weights()

    def plot_accuracy(self):
        plt.plot(self.accuracies)
        plt.title('Dokładność (Accuracy) w miarę epok')
        plt.xlabel('Epoka')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_weights(self):
        # Wyświetlanie wag warstwy ukrytej i wyjściowej
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(self.weight_history_hidden[-1], cmap='viridis')
        ax1.set_title('Hidden Layer Weights')

        ax2.imshow(self.weight_history_output[-1], cmap='viridis')
        ax2.set_title('Output Layer Weights')

        plt.show()