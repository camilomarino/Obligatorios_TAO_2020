# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim

device = 'cuda'

def to_np(x):
    return x.detach().cpu().numpy()

def adam_pgd(D: np.ndarray, 
            X: np.ndarray, 
            lr: float = 1e-2,
            betas=(0.99, 0.999),
            max_iter = 70_000,
            A0: np.ndarray = None,
            verbose: bool = True) -> np.ndarray:
    
    X = torch.tensor(X, device=device, dtype=torch.float32)
    D = torch.tensor(D, device=device, dtype=torch.float32)
    f = lambda A: torch.norm(X-D@A)**2
    if A0 is None:
        A = torch.zeros((D.shape[1], X.shape[1]), device=device, 
                         dtype=torch.float32, requires_grad=True)
    else:
        A = torch.tensor(A0, device=device, dtype=torch.float32,
                          requires_grad=True)
    
    optimizer = optim.Adam([A], lr=lr, betas=betas)
    
    for k in range(max_iter):
        optimizer.zero_grad()
        loss = f(A)
        loss.backward()
        optimizer.step()
        if k%1000==0 and verbose:
            print(f'[iter:{k}]\t{float(loss):.2f}')
    
        with torch.no_grad():
            A.clip_(0, None)

    return to_np(A), float(loss)
