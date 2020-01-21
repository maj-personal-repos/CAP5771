from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# simple perceptron artificial neural network

# modeling a boolean function
# linearly separable, so perceptron works
X = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0]]
y = [-1, 1, 1, 1, -1, -1, 1, -1]

# lets plot it to see

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, set in enumerate(X):
    color = 'r' if y[idx] == -1 else 'b'
    ax.scatter(X[idx][0], X[idx][1], X[idx][2], c=color, marker='o')

plt.show()
plt.scatt

perceptron = Perceptron(alpha=1e-5, random_state=1)

perceptron.fit(X, y)

print(perceptron.coef_)
print(perceptron.intercept_)
print(perceptron.predict(X))
print(perceptron.score(X, y))

# TODO plot decision boundary


# multilayer ANN, good for more complex, non-linear data
# XOR example

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4,), random_state=1)

mlp.fit(X, y)


print(mlp.predict(X))
print(mlp.score(X, y))

# TODO plot decision boundary
