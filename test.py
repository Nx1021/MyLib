import time
import numpy as np

a = np.array([])
b = []

start = time.time()
for i in range(100000):
    a = np.append(a, i)
print("Numpy append: ", time.time() - start)

start = time.time()
for i in range(100000):
    b.append(i)
print("List append: ", time.time() - start)