import numpy as np
import math
from scipy import stats

# Dissimilarity
# Euclidean Distance

def euclidean_distan(x, y):
    return np.linalg.norm(x - y)


p1 = np.array([0, 2])
p2 = np.array([2, 0])
p3 = np.array([3, 1])
p4 = np.array([5, 1])

print("Eucledean Distance: " + str(euclidean_distan(p1, p1)))


# Minkoswski Distance

def minkowski_distanc(x, y, r):
    return np.linalg.norm(x - y, r)


print("Hamming Distance: " + str(minkowski_distanc(p1, p2, 1)))
print("Euclidean Distance: " + str(minkowski_distanc(p1, p2, 2)))
print("Supremum Distance: " + str(minkowski_distanc(p1, p2, math.inf)))

# Similarity
# Similarity Measures for Binary Data

x = [1,0,0,0,0,0,0,0,0,0]
y = [0,0,0,0,0,0,1,0,0,1]


# Simple Matching Coefficient

def smc(x, y):
    if len(x) != len(y):
        print("vectors must be of the same length!")
        return

    f01 = 0
    f10 = 0
    f00 = 0
    f11 = 0

    for x0, y0 in zip(x, y):
        if x0 == 1:
            if x0 == y0:
                f11 += 1
            else:
                f10 += 1
        else:
            if x0 == y0:
                f00 += 1
            else:
                f01 += 1

    return (f11 + f00)/(f01+f10+f11+f00)



# Jaccard Coefficient
def jaccard(x, y):
    if len(x) != len(y):
        print("vectors must be of the same length!")
        return

    f01 = 0
    f10 = 0
    f00 = 0
    f11 = 0

    for x0, y0 in zip(x, y):
        if x0 == 1:
            if x0 == y0:
                f11 += 1
            else:
                f10 += 1
        else:
            if x0 == y0:
                f00 += 1
            else:
                f01 += 1

    return (f11) / (f01 + f10 + f11)


print("SMC similarity: " + str(smc(x, y)))
print("Jaccard similarity: " + str(jaccard(x, y)))

# Cosine Similarity

x = [3, 2, 0, 5, 0, 0, 0, 2, 0, 0]
y = [1, 0, 0, 0, 0, 0, 0, 1, 0, 2]


def cos_sim(x, y):
    x = np.array(x)
    y = np.array(y)

    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))


print("Cosine similarity: " + str(cos_sim(x, y)))

# Correlation

x = [-3, 6, 0, 3, -6]
y = [1, -2, 0, -1, 2]

print(stats.pearsonr(x, y)[0])

x = [3, 6, 0, 3, 6]
y = [1, 2, 0, 1, 2]

print(stats.pearsonr(x, y)[0])