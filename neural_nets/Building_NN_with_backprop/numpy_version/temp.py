import numpy as np

A = np.array([[1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]])
B = np.argmax(A, axis=1)

print(B)