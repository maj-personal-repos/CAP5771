import numpy as np


# create some vectors

A = np.array([1, 2])
B = np.array([2, 1])

# addition

print(A + B)

# multiplication by a scalar

print(3 * A)

# dot product

print(np.dot(A, B))

# element wise multiplication

print(A * B)

# orthogonality test

C = np.array([0, 1])
D = np.array([1, 0])

if np.dot(C, D) == 0:
    print("The vectors " + str(C) + " and " + str(D) + " are orthogonal.")
else:
    print("The vectors " + str(C) + " and " + str(D) + " are not orthogonal.")
