#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:24:34 2020

@author: camilo
"""

import cvxpy as cp
import numpy as np
from utils import get_Pi, get_qi, get_ri, get_constrain_i, get_constrain_i_value
from utils import P0, q0, r0, N, f

l  = cp.Variable(N)
t  = cp.Variable((1,1))

def get_Pqr(P0, q0, r0, l, N):
    P = P0.copy()
    q = q0.copy()
    r = r0.copy()
    # import ipdb; ipdb.set_trace()
    for i in range(1, N+1):
        P += get_Pi(i, N) * l[i-1]
        q += get_qi(i, N) * l[i-1]
        r += get_ri(i) * l[i-1]
    
    return P, q, r
    
P, q, r = get_Pqr(P0, q0, r0, l, N)

X = cp.Variable((N+1,N+1), PSD=True)

# Una hora y media buscando el error y era que habia puesto Minimize 
# y no Maximize!!!!
objective = cp.Maximize(t)

constraints = [X == cp.bmat([[P, q], [q.T, r - t]])]
prob = cp.Problem(objective, constraints)

result = prob.solve()

print('d_pb = ', result)
