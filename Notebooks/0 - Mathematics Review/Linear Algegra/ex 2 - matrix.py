import numpy as np
from scipy import linalg

# create some matrices

b = np.array([(1 + 5j, 2j, 3j), (4j, 5j, 6j)])

A = np.matrix(np.random.random((2,2)))
B = np.asmatrix(b)
C = np.mat(np.random.random((10,5)))
D = np.mat([[3,4], [5,6]])

print(A)
print(B)
print(C)
print(D)

# inverse

print(A.I)

# transpose

print(A.transpose())

# trace

print(A.trace())

# norm

print(linalg.norm(A))
print(linalg.norm(A, 1))
print(linalg.norm(A, np.inf))


# rank

print(np.linalg.matrix_rank(C))

# determinant

print(linalg.det(A))

# solving linear problems

print(linalg.solve(A, b))


# matrix addition

print(np.add(A, D))

# matrix subtraction

print(np.subtract(A, D))

# matrix division

print(np.divide(A, D))


# matrix multiplication

print(np.multiply(A, D))
print(np.dot(A, D))

# martix exponential

print(linalg.expm(A))

# matrix decompositions

# eigen values and vectors

la, v = linalg.eig(A)

l1, l2 = la

print(l1)
print(l2)
print(v[:, 0])
print(v[:, 1])


# singular value decomposition

U, s, Vh = linalg.svd(D)
M, N = D.shape
Sig = linalg.diagsvd(s, M, N)
print(Sig)
