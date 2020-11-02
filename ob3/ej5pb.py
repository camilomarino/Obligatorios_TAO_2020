import numpy as np

import itertools
from utils import f, N

f_pb = np.inf
for x in itertools.product([-1, 1], repeat=N):
    x = np.array(x)
    f_value = float(f(x).value)
    if f_value < f_pb:
        f_pb = f_value
        x_pb = x
    
    
print('xpb* = ', x_pb)
print('fpb* = ', f(x_pb).value)