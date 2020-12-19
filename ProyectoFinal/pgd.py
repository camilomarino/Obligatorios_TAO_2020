#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:34:17 2020

@author: camilo
"""
import numpy as np
from time import time
from tqdm import tqdm

def pgd(D:np.ndarray, 
        X:np.ndarray, 
        max_iter = 10000,
        min_consecutive_diff = 1e-5,
        weight_decay: float = 0,
        verbose: bool = True) -> np.ndarray:
    X = X.reshape((X.shape[0], 1))
    ti = time()
    if X.shape[1] !=1:
        raise AttributeError('X debe ser una matriz columna')
    p = D.T@X
    Q = D.T@D
    lr = 1/np.linalg.norm(Q, ord=2)
    A_shape = (D.shape[1], X.shape[1])
    Ak = np.ones(A_shape) / np.prod(A_shape)
    for i in tqdm(range(max_iter)):
        A_ant = Ak
        Ak = Ak - lr * ( 2*(Q@Ak-p) + weight_decay*np.sign(Ak))
        Ak[Ak<0] = 0
        # if Ak.sum()>1:
        #     Ak = Ak / Ak.sum()
            
        if np.linalg.norm(Ak-A_ant)<min_consecutive_diff:
            break
    cost = np.linalg.norm(X - D@Ak, ord=2)**2 
    if verbose: print(f'Cantidad de iteraciones: {i}\nTiempo: {time()-ti:.2f}')
    return Ak.reshape((Ak.shape[0],)), cost
