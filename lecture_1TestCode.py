##Einops
from einops import einsum
import numpy as np
import torch as torch

x = np.array([0, 1, 10])

y = einsum(x, 'i -> i')
print(y)

y = einsum(x, 'i -> ')
print(y)

y = einsum(x, x, 'i, i -> i')
print(y)

