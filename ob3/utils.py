import cvxpy as cp
import numpy as np

def get_Pi(i: int, N: int):
    j = i - 1
    if i>N:
        raise AttributeError("i es mayor a N")
    Pi = np.zeros((N, N))
    Pi[j, j] = 1
    return Pi

def get_qi(i: int, N: int):
    return np.zeros((N,1))

def get_ri(i: int):
    return -1

def get_constrain_i(x, i, N):
    Pi = get_Pi(i, N)
    qi = get_qi(i, N)
    ri = get_ri(i)    
    return cp.quad_form(x, Pi) + 2*qi.T + ri <= 0

def get_constrain_i_value(x, i, N):
    Pi = get_Pi(i, N)
    qi = get_qi(i, N)
    ri = get_ri(i)    
    return (cp.quad_form(x, Pi) + 2*qi.T + ri).value

P0 = np.loadtxt('data/P.asc')
q0 = np.loadtxt('data/q.asc').reshape((-1,1))
r0 = np.loadtxt('data/r.asc').reshape((1,1))

N = q0.shape[0]

f = lambda x:cp.quad_form(x, P0) + 2*q0.T@x + r0

