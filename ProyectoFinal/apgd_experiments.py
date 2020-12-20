#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:39:07 2020

@author: camilo
"""

import numpy as np
from time import time
from tqdm import tqdm
import torch
device = 'cuda'

from torch import optim

#%%

X = np.load('X.npy')#[:, 0]
D = np.load('D.npy')
A_opt = np.load('A_opt.npy')
#%%
X = torch.tensor(X, device=device, dtype=torch.float32)
D = torch.tensor(D, device=device, dtype=torch.float32)
A_opt = torch.tensor(A_opt, device=device, dtype=torch.float32)
A = torch.zeros((D.shape[1], X.shape[1]), device=device, dtype=torch.float32, requires_grad=True)
p = D.T@X
Q = D.T@D
lr = 5e-3#100/torch.norm(Q)

optimizer = optim.Adam([A], lr=lr)
f = lambda A: torch.norm(X-D@A)
#%%
for _ in range(100000):
    optimizer.zero_grad()
    loss = f(A)
    loss.backward()
    optimizer.step()
    print(float(loss))
    with torch.no_grad():
        A.clip_(0, None)

#%%
def to_np(x):
    return x.detach().cpu().numpy()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(to_np(A[:,0]), label='Adam-PGD')
plt.plot(to_np(A_opt), label='NNLS', alpha=0.5)
plt.legend()