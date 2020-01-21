import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def visualize_iris():
    iris = load_iris()
    # parameters
    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Plot
        plt.subplot(2, 3, pairidx + 1)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Iris Dataset Visualization")
    # plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    return plt