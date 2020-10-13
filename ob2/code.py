#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:10:41 2020

@author: camilo
"""

import numpy as np

#%%

A = np.loadtxt('A.txt')
B = np.loadtxt('B.txt')
C = np.loadtxt('C.txt')
alpha = np.loadtxt('alpha.txt')
beta = np.loadtxt('beta.txt')
gamma = np.loadtxt('gamma.txt')

#%%
# Tamano de n
_, nA = A.shape
_, nB = B.shape
_, nC = C.shape

# Sanity Check
if nA==nB==nC:
    n = nA
    del nA, nB, nC
else:
    raise ValueError

#%%
# Calculo de optimo

def calc_x_optim(A, B, C, alpha, beta, gamma):
    x_opt = np.linalg.inv(A.T@A + B.T@B + C.T@C) @ (
                    A.T@alpha + B.T@beta + C.T@gamma)
    return x_opt

x_opt = calc_x_optim(A, B, C, alpha, beta, gamma)

#%%
# Calculo de las matrices D, H y S

def calc_D(A, B, C, n):
    D = np.zeros((3*n, 3*n))
    D[0:n, 0:n] = A
    D[n:2*n, n:2*n] = B
    D[2*n:3*n, 2*n:3*n] = C
    return D

def calc_H(n):
    H = np.zeros((2*n, 3*n))
    H[:, 0:n] = 1
    H[0:n, n:2*n] = -1
    H[n:2*n, 2*n:3*n] = -1
    return H

def calc_S(alpha, beta, gamma):
    S = np.concatenate((alpha, beta, gamma))
    return S

D = calc_D(A, B, C, n)    
H = calc_H(n)
S = calc_S(alpha, beta, gamma)


#%%
def calc_lambda_optim(D, S, H):
    lambda_optim = np.linalg.inv(H@np.linalg.inv(D.T@D)@H.T) @ H @ np.linalg.inv(D) @ S
    return lambda_optim

def calc_w_optim(D, S, H):
    w_optim = np.linalg.inv(D)@S - np.linalg.inv(D.T@D)@H.T@calc_lamda_optim(D, S, H)
    return w_optim

def calc_x_from_w(w):
    xyz = np.reshape(w, (3,-1))
    x = np.mean(xyz, axis=0)
    return x

lambda_optim = calc_lambda_optim(D, S, H)
w_optim = calc_w_optim(D, S, H)
x_optim_from_w = calc_x_from_w(w_optim)
#%%
def calc_tau0(D):
    tau0 = 1/np.linalg.norm(D, ord=2)
    return tau0
tau0 = calc_tau0(D)

#%%
def calc_error(wk, wk_1):
    error = np.linalg.norm(wk - wk_1) / np.linalg.norm(wk)
    return error


