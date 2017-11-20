from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# simple perceptron artificial neural network

# modeling a boolean function
# linearly separable, so perceptron works
X = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0]]
y = [-1, 1, 1, 1, -1, -1, 1, -1]

perceptron = Perceptron(alpha=1e-5, random_state=1)

perceptron.fit(X, y)

print(perceptron.coef_)
print(perceptron.intercept_)
# print(perceptron.predict(X))
# print(perceptron.score(X, y))

# TODO plot decision boundary


# multilayer ANN, good for more complex, non-linear data
# XOR example

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4,), random_state=1)

mlp.fit(X, y)

# print(mlp.predict(X))

# TODO plot decision boundary
