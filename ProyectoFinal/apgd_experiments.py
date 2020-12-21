#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:39:07 2020

@author: camilo
"""

import numpy as np
from tqdm import tqdm
import torch
device = 'cuda'

from torch import optim

#%%

X = np.load('X.npy')
D = np.load('D.npy')
A_opt = np.load('A_opt.npy')
X = torch.tensor(X, device=device, dtype=torch.float32)
D = torch.tensor(D, device=device, dtype=torch.float32)
A_opt = torch.tensor(A_opt, device=device, dtype=torch.float32)

#%%
A = torch.zeros((D.shape[1], X.shape[1]), device=device, dtype=torch.float32, requires_grad=True)
p = D.T@X
Q = D.T@D
lr = 10e-3#100/torch.norm(Q)
#%%
optimizer = optim.Adam([A], lr=lr, betas=(0.99, 0.999))
f = lambda A: torch.norm(X-D@A)
#%%
MAX_ITER = 100_000
for k in range(MAX_ITER):
    optimizer.zero_grad()
    loss = f(A)
    loss.backward()
    optimizer.step()
    if k%1000==0:
        print(f'{k}\t', float(loss), '\t')

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

#%%
cost_total = 0
for i, (x, a) in enumerate(zip(X.T, A.T)):
    cost = torch.norm(x-D@a)
    print(f'[{i}] Costo: {cost}')
    cost_total += cost