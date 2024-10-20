from MLP import *

# Training data (4 patterns)
X = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Model creation
mlp = MultilayerPerceptron(input_size=4, hidden_size=2, output_size=4, learning_rate=0.1)
print(mlp.w_hidden)
print(mlp.w_output)
# Network training
mlp.train(X, y)

# Testing network
for i in range(len(X)):
    output = mlp.forward(X[i])
    output_formatted = ', '.join([f'{o:.3f}' for o in output])
    print(f'Input: {X[i]} Predicted output: {output_formatted}')