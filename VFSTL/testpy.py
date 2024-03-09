import torch as th
import numpy as np
a = th.tensor([[1,2,3,4], [1,2,3,4]]).T
print(a)
print(th.max(a, 0).values)
print([th.tensor([10, 10])]*10)

print(th.count_nonzero(th.tensor([False, False, 1])))