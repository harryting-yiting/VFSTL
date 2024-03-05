import numpy as np

a = np.array([[1, 2, 3], [1,2,3]])
b = np.array([[4, 5, 6, 7],[4,5,6,7]])

c = np.concatenate((a, b),1)

print(c)

import torch.nn as nn
d = nn.functional.one_hot(a)
print(d)