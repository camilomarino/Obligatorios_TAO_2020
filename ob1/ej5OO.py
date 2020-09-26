#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:57 2020

@author: camilo
"""


    
#%%
import numpy as np
from typing import List
import matplotlib.pyplot as plt
dtype = np.float64
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 30
plt.rcParams['figure.figsize'] = [16, 10]

class FunctionToMinimize():
    def __init__(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
    def __call__(self, x):
        return self.forward(x)


class MinSquareRoot(FunctionToMinimize):    
    def __init__(self, A, b):
        self.A = A
        self.b = b
    
    def forward(self, x):
        self._x_is_valid_shape(x)
        return np.linalg.norm(self.A@x-self.b)**2
    
    def backward(self, x):
        return 2*self.A.T@(self.A@x-self.b)
    
    def min(self):
        return np.linalg.inv((self.A.T@self.A))@self.A.T@self.b
    
    def _x_is_valid_shape(self, x) -> None:
        if x.shape != (self.A.shape[1], 1):
            raise AttributeError("x debe se un vector columna")
            
            
            
class Optimizer:
    '''
    type_step:
        constant: kwargs:
                    step
        
        decreasing: kwargs:
                    constant
        
        line_search: kwargs:
                        n_points: numero de puntos de la grilla
                        long: largo de la grilla
        armijo: kwargs:
                    sigma
                    beta
                    s
                    
    D: Matriz D que premultiplica al gradiente
    
    '''
    def __init__(self, f, type_step, **kwargs):
        self.f = f
        self.type = type_step
        self.step_driver = self._get_step_instance(f, type_step, **kwargs)
        self.k = 0
        self.xk = self._get_x0(x0=kwargs.get('x0'))
        if kwargs.get('D') is not None:
            self.D = kwargs.get('D')
        else :
            self.D = np.eye(2)
        
    def get_xk(self):
        return self.xk
    
    def get_k(self):
        return self.k
    
    def info(self):
        return {'k':self.k, 'xk':self.xk, 'type':self.type}
    
    def step(self):
        self.k += 1
        self.xk = self.step_driver.next_xk(xk=self.xk, k=self.k, D=self.D)
        return self.k, self.xk
    
    def _get_step_instance(self, f, type_step, **kwargs):
        if type_step=='constant' or type_step=='constante':
            return ConstantStep(f, **kwargs)
        elif type_step=='decreasing' or type_step=='decreciente':
            return DecreasingStep(f, **kwargs)
        elif type_step=='line_search':
            return LineSearch(f, **kwargs)
        elif type_step=='armijo':
            return Armijo(f, **kwargs)
        else:
            raise AttributeError
  
    def _get_x0(self, x0=None):
        if x0 is None:
            return np.zeros((2,1), dtype=dtype)
        else:
            return x0
        
        
        
    
class Step():
    def __init__(self):
        raise NotImplementedError
    def next_step():
        raise NotImplementedError
        
class ConstantStep(Step):
    def __init__(self, f, step: float = 0.01, **kwargs):
        self.step = step
        self.f = f
        
    def next_xk(self, xk, k, D):
        xk = xk - self.step * D@f.backward(xk)
        return xk
   
    
class DecreasingStep(Step):
    def __init__(self, f, constant: float = 0.01, **kwargs):
        self.constant = constant
        
    def next_xk(self, xk, k, D) -> List[float]:
        step = self.constant/k
        xk = xk - step * D@f.backward(xk)
        return xk
        
        
class LineSearch(Step):
    def __init__(self, f, n_points=1000, long=1, **kwargs):
        self.n_points = n_points
        self.long = long
        self.f = f
        self.steps = self._get_steps()
        
    def next_xk(self, xk, k, D):
        w = D@f.backward(xk)
        xk_candidatos = np.tile(xk, self.steps.size) - self.steps * w
        #import ipdb; ipdb.set_trace()
        fxk_candidatos = list()
        for k in range(xk_candidatos.shape[1]):
            fxk_candidatos.append(f.forward(xk_candidatos[:,k:k+1]))
        fxk_candidatos = np.array(fxk_candidatos)
        
        arg_min = fxk_candidatos.argmin()
        return xk_candidatos[:, arg_min:arg_min+1]
    
    def _get_steps(self):
        return np.linspace(0, self.long, self.n_points)
    
    
class Armijo(Step):
    def __init__(self, f, sigma=0.1, beta=0.5, s=0.1, **kwargs):
        self.f = f
        self.sigma = sigma
        self.beta = beta
        self.s = s
    
    def next_xk(self, xk, k, D):
        #import ipdb; ipdb.set_trace()
        grad_f = self.f.backward(xk)
        dk = -D@grad_f
        alfa = self.s
        while True:            
            desigualdad_izq = (self.f(xk) -
                               self.f(xk+alfa*self.s*dk))
            desigualdad_der = -self.sigma*alfa*self.s*grad_f.T@dk
            if desigualdad_izq >= desigualdad_der:
                break
            alfa *= self.beta
        xk = xk+alfa*self.s*dk
        return xk
        

class ErrorToOptim:
    def __init__(self, x_sol):
        self.x_sol = x_sol
    def __call__(self, xk):
        return np.linalg.norm(self.x_sol-xk)/np.linalg.norm(self.x_sol)
    

class Error:
    def __init__(self):
        pass
    def __call__(self, xk, xk_1):
        return np.linalg.norm(xk-xk_1)/np.linalg.norm(xk_1)
    
A = np.array([[-4.100000000000000000e+01, 2.000000000000000000e+01],
            [-4.600000000000000000e+01, -8.000000000000000000e+00],
            [-5.000000000000000000e+00, -3.300000000000000000e+01],
            [-5.500000000000000000e+01, 1.000000000000000000e+00],
            [-5.500000000000000000e+01, -6.000000000000000000e+00]])

b = np.array([8.000000000000000000e+00 , 5.000000000000000000e+00, 
            -3.000000000000000000e+00, 1.000000000000000000e+01, 
            4.000000000000000000e+00])
b = b[:, np.newaxis]



f = MinSquareRoot(A, b)
error_to_optim = ErrorToOptim(f.min())
error = Error()


from tqdm import tqdm
def calc_trajectory(opt, error, iteraciones=1000, min_error=0.0001):
    k, x, e = [opt.k], [opt.xk], [error_to_optim(opt.xk)]
    
    for iter in range(1,iteraciones):
        i, xk = opt.step()
        k.append(i)
        x.append(xk)
        e.append(error_to_optim(opt.xk))
        # if iter>3:
        #     print(error(x[-1], x[-2]))
        if iter>3 and error(x[-1], x[-2])<min_error:
            break
        
        
        
    x = np.array(x)[...,0].T 
    k = np.array(k)
    return k, x, e
        
def plot_trayectory(x, title, annotate=True, k=None):
    plt.scatter(x[0], x[1])
    if annotate: 
        for i in range(k.shape[0]):
            if i%15==0:
                plt.annotate(k[i], (x[0,i], x[1,i]))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.savefig(title+'png')

def plot_error(e, title=None):
    plt.plot(e, label=title)
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    if title is not None: plt.title(title)
    plt.savefig(title+'png')
    
def print_info(k,x,e, tipo=""):
    x_ = x[0,-1]
    y_ = x[1,-1]
    k_ = k[-1]
    e_ = e[-1]
    print('-'*55)
    print(tipo)
    print(f'El punto optimpo es ({x_:.5f},{y_:.5f})')
    print(f'Con {k_} iteraciones')
    print(f'Un error final de {e_:.5f}')
    print(f'Un error relativo respecto al paso anterior de {error(x[:,-1], x[:,-2])}')
x0 = np.full((2,1), 0)

#%%    
plt.close('all')
step = 1/(2*np.linalg.norm(A,2)**2)
opt = Optimizer(f, 'constante', step=step, x0=x0)
k, x, e = calc_trajectory(opt, error)
plt.figure()
plot_trayectory(x, title='Trayectoria Paso constante', k=k)
plt.figure()
plot_error(e, title='Paso constante')
print_info(k,x,e,'Paso constante' )

#%%
opt = Optimizer(f, 'decreciente', constant=0.001, x0=x0)
k, x, e = calc_trajectory(opt, error)
plt.figure()
plot_trayectory(x, title='Trayectoria Paso decreciente', k=k)
plt.figure()
plot_error(e, title='Paso decreciente')
print_info(k,x,e,'Paso constante' )

#%%
opt = Optimizer(f,'line_search', n_points=100, long=0.001, x0=x0)
k, x, e = calc_trajectory(opt, error)
plt.figure()
plot_trayectory(x, title='Trayectoria Line Search', k=k)
plt.figure()
plot_error(e, title='Line Search')
print_info(k,x,e,'Line Search' )

#%%
plt.close('all')
opt = Optimizer(f,'armijo', sigma=0.1, beta=0.5, s=0.01, x0=x0)
k, x, e = calc_trajectory(opt, error)
plt.figure()
plot_trayectory(x, title='Trayectoria Armijo', k=k)
plt.figure()
plot_error(e, title='Armijo')
print_info(k,x,e,'Armijo' )


#%%
# Grid search sobre hiperparametros de Armijo
plt.close('all')
grid = list()
for sigma in list(np.linspace(0,0.3,20)):
    for beta in list(np.linspace(0.3,0.7,20)):
        for s in list(np.logspace(-1,-5, 20)):
            opt = Optimizer(f,'armijo', sigma=sigma, beta=beta, s=s, x0=x0)
            k, x, e = calc_trajectory(opt, error, iteraciones=300)
            
            grid.append((sigma, beta, s, k[-1]))

#%%
ite = [i for _,_,_,i in grid]
ite = np.array(ite)
amin = ite.argmin()
sigma, beta, s, _ = grid[amin]
print(grid[amin]) #4643
#%%
plt.close('all')
opt = Optimizer(f,'armijo', sigma=sigma, beta=beta, s=s, x0=x0)
k, x, e = calc_trajectory(opt, error)
plt.figure()
plot_trayectory(x, title='Trayectoria Armijo Optima', k=k)
plt.figure()
plot_error(e, title='Armijo Optimo')
print_info(k,x,e,'Armijo Optimo' )