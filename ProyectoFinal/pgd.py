#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:34:17 2020

@author: camilo
"""
import numpy as np
import torch
from time import time
from tqdm import tqdm
device = 'cuda'

def pgd(D:np.ndarray, 
        X:np.ndarray, 
        max_iter = 10000,
        min_consecutive_diff = 1e-5,
        weight_decay: float = 0,
        Ak: np.ndarray = None,
        verbose: bool = True) -> np.ndarray:
    #X = X.reshape((X.shape[0], 1))
    ti = time()
    #if X.shape[1] !=1:
        #raise AttributeError('X debe ser una matriz columna')
    p = D.T@X
    Q = D.T@D
    lr = 1/np.linalg.norm(Q, ord=2)
    A_shape = (D.shape[1], X.shape[1])
    if Ak is None:
        Ak = np.ones(A_shape) / np.prod(A_shape)
    else:
        Ak = Ak.reshape(A_shape)
    losses = list()
    for i in tqdm(range(max_iter)):
        losses.append(np.linalg.norm(X - D@Ak, ord=2)**2)
        A_ant = Ak
        Ak = Ak - lr * ( 2*(Q@Ak-p) + weight_decay*np.sign(Ak))
        Ak[Ak<0] = 0
        
        # if Ak.sum()>1:
        #     Ak = Ak / Ak.sum()
            
        if np.linalg.norm(Ak-A_ant)<min_consecutive_diff:
            break
    cost = np.linalg.norm(X - D@Ak, ord=2)
    if verbose: print(f'Cantidad de iteraciones: {i}\nTiempo: {time()-ti:.2f}')
    return Ak, cost, losses


# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim

device = 'cuda'

def to_np(x):
    return x.detach().cpu().numpy()

def pgd_torch(D: np.ndarray, 
            X: np.ndarray, 
            #lr: float = 1e-2,
            #betas=(0.99, 0.999),
            max_iter = 70_000,
            A0: np.ndarray = None,
            mask: np.ndarray = None,
            n_mask: int = None,
            verbose: bool = True) -> np.ndarray:
    
    X = torch.tensor(X, device=device, dtype=torch.float32)
    D = torch.tensor(D, device=device, dtype=torch.float32)
    
    #p = D.T@X
    Q = D.T@D
    lr = 1/torch.norm(Q)
    
    f = lambda A: torch.norm(X-D@A)**2
    if mask is not None: mask = torch.tensor(mask, device=device)
    
    if A0 is None:
        A = torch.zeros((D.shape[1], X.shape[1]), device=device, 
                         dtype=torch.float32, requires_grad=True)
    else:
        A = torch.tensor(A0, device=device, dtype=torch.float32,
                          requires_grad=True)
    
    optimizer = optim.SGD([A], lr=lr)
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
