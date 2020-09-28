#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:57 2020

@author: camilo
"""


    
#%%
import numpy as np
import matplotlib.pyplot as plt
from Function import MinSquareRoot
from Optimizer import Optimizer
from Error import ErrorToOptim, Error
from utils import calc_trajectory, plot_trayectory, plot_error, print_info

dtype = np.float64
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 30
plt.rcParams['figure.figsize'] = [16, 10]
            
    
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

x0 = np.full((2,1), 0)

#%%    
plt.close('all')
step = 1/(2*np.linalg.norm(A,2)**2)
opt = Optimizer(f, 'constante', step=step, x0=x0)
k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria Paso constante', k=k)
plt.figure()
plot_error(e, title='Paso constante')
print_info(k,x,e,error,'Paso constante' )

#%%
opt = Optimizer(f, 'decreciente', constant=0.001, x0=x0)
k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria Paso decreciente', k=k)
plt.figure()
plot_error(e, title='Paso decreciente')
print_info(k,x,e,error,'Paso constante' )

#%%
opt = Optimizer(f,'line_search', n_points=100, long=0.001, x0=x0)
k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria Line Search', k=k)
plt.figure()
plot_error(e, title='Line Search')
print_info(k,x,e,error,'Line Search' )

#%%
plt.close('all')
opt = Optimizer(f,'armijo', sigma=0.1, beta=0.5, s=0.01, x0=x0)
k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria Armijo', k=k)
plt.figure()
plot_error(e, title='Armijo')
print_info(k,x,e,error,'Armijo' )


#%%
# Grid search sobre hiperparametros de Armijo
plt.close('all')
grid = list()
for sigma in list(np.linspace(0,0.3,20)):
    for beta in list(np.linspace(0.3,0.7,20)):
        for s in list(np.logspace(-1,-5, 20)):
            opt = Optimizer(f,'armijo', sigma=sigma, beta=beta, s=s, x0=x0)
            k, x, e = calc_trajectory(opt, error, error_to_optim, iteraciones=300)
            
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
k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria Armijo Optima', k=k)
plt.figure()
plot_error(e, title='Armijo Optimo')
print_info(k,x,e,error,'Armijo Optimo' )