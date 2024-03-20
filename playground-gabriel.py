"""
import numpy as np 
import pandas as pd

x = np.array([[1],[2],[3],[4],[5]])

print("Vector x: \n",x)

sin_x = np.sin(x)

print("Vector sin_x: \n",sin_x)

exp_x = np.exp(x)

print("Vector exp_x: \n",exp_x)

x_squared = np.square(x)

print(x.size)

RMSE = np.sqrt(np.mean(np.square(x)))

print("RMSE: ", RMSE)

for i in range(10):
    print(i)

import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=4,shuffle=True)
kf.get_n_splits(X)
print(kf)


for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

"""