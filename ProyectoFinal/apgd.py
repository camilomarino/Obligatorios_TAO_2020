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
            mask: np.ndarray = None,
            n_mask: int = None,
            verbose: bool = True) -> np.ndarray:
    
    X = torch.tensor(X, device=device, dtype=torch.float32)
    D = torch.tensor(D, device=device, dtype=torch.float32)
    f = lambda A: torch.norm(X-D@A)**2
    if mask is not None: mask = torch.tensor(mask, device=device)
    
    if A0 is None:
        A = torch.zeros((D.shape[1], X.shape[1]), device=device, 
                         dtype=torch.float32, requires_grad=True)
    else:
        A = torch.tensor(A0, device=device, dtype=torch.float32,
                          requires_grad=True)
    
    optimizer = optim.Adam([A], lr=lr, betas=betas)
    losses = list()
    for k in range(max_iter):
        optimizer.zero_grad()
        loss = f(A)
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        if k%1000==0 and verbose:
            print(f'[iter:{k}]\t{float(loss):.2f}')
    
        with torch.no_grad():
            A.clip_(0, None)
            if mask is not None:
                eps = 1e-10
                for i in range(n_mask):
                    A[mask==i] /= (A[mask==i].sum(dim=0) + eps)
                    

    return to_np(A), float(loss), losses
