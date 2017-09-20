# TODO Comment

import numpy as np
import pandas as pd


def flip(word_size, trials):
    return np.random.randint(0, 2, size=(trials, word_size))


def compute_mean(data):
    df = pd.DataFrame(data)
    tails = df.sum(axis=1)
    return tails.mean()


def compute_variance(data):
    df = pd.DataFrame(data)
    tails = df.sum(axis=1)
    return tails.var()


if __name__ == "__main__":
    experiment_data = flip(4, 10000)
    mean = compute_mean(experiment_data)
    variance = compute_variance(experiment_data)
    print("The  expected value is: " + str(mean))
    print("The variance is: " + str(variance))
