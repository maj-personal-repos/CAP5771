import numpy as np
import matplotlib.pyplot as plt



def flip(word_size, trials):
    return np.random.randint(0, 2, size=(trials, word_size))


if __name__ == "__main__":
    experiment_data = flip(4, 10000)
    number_of_tails = np.sum(experiment_data, axis=1)
    n, bins, patches = plt.hist(number_of_tails, 5, normed=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Number of Tails')
    plt.ylabel('Probability')
    plt.title('Probabilty Mass Function for number of Tails')
    plt.axis([0, 4, 0, 4000])
    plt.grid(True)
    plt.show()
