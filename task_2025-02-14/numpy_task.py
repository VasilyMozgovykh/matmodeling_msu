import numpy as np


print("Введите n")
n = int(input())
arr = np.flipud(np.diag(np.arange(n-1, 0, -1, dtype=np.uint64), 1))
print(arr)
