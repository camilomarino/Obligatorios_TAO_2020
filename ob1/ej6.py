#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:02:34 2020

@author: camilo
"""

import numpy as np
import matplotlib.pyplot as plt
from Function import CustomFunction
from Optimizer import Optimizer
from Error import ErrorToOptim, Error, ErrorToPreviousSample
from utils import calc_trajectory, plot_trayectory, plot_error, print_info
from utils import plot_f_evolution
plt.rcParams['font.size'] = 25


f_expr = lambda x,y: 5*x**2 + 5*y**2 + 5*x - 3*y - 6*x*y +5/4
grad_f = lambda x,y: (10*x + 5 - 6*y,  10*y - 3 - 6*x) 

f = CustomFunction(f_expr, grad_f)
error_to_optim = ErrorToOptim(np.array([[-0.5],[0]]))
error = ErrorToPreviousSample()

x0 = np.full((2,1), 0)

constant = 1

def proyection_to_the_circle(x, R=0.25):
    norm = np.linalg.norm(x)
    if norm>R:
        x = R*x/norm
    return x
       
def proyection_out_of_the_circle(x, R=1):
    norm = np.linalg.norm(x)
    if norm<R:
        x = x*R/norm
    return x 
    

def aux():
    fx = -0.1
    fy = -0.1
    f = lambda x,y: 5*x**2 + 5*y**2 + 5*x - 3*y - 6*x*y +5/4
    f_ = '5x^2 + 5y^2 + 5x - 3y - 6xy +5/4'
    
    x = np.linspace(-1, 1., 1000)
    y = np.linspace(-1, 1, 1000)
    
    xx, yy = np.meshgrid(x, y)
    
    z = f(xx, yy)
    
    z[np.isnan(z)] = 1000
        
    
    levels = np.sort(np.append(np.linspace(-15, 25, 50),0))
    
    # plt.figure(figsize=(10,10))
    # plt.title(f'Curvas de nivel de $f(x,y) = {f_}$')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.grid(True)
    contours = plt.contour(xx, yy, z, levels, linewidths=2, color='black')
    plt.clabel(contours, inline=True, fontsize=8)
    
    
    
    x = np.linspace(-1.1, 1.1, 100)
    y = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 
    plt.contour(X,Y,F,[1**2])
    plt.show()
#%%
def plots_info(opt, name):
    plt.close('all')
    
    k, x, e, fxk = calc_trajectory(opt, error, error_to_optim, more_data=True)
    #import ipdb; ipdb.set_trace()
    plt.figure(figsize=(13,13))
    aux()
    plot_trayectory(x, title=f'Trayectoria {name}', k=k, with_lines=True,
                                   step_numbers=1000000)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.figure()
    plot_error(e, title=f'Error {name}')
    print_info(k,x,e,error,f'{name}')
    plot_f_evolution(fxk, title=f'Evolucion de f {name}')

#%%
opt = Optimizer(f, 'PGD_DecreasingStep', constant=constant, 
                    proyection=proyection_to_the_circle, x0=x0)
name = 'PGD Paso Decreciente Convexo'
plots_info(opt, name)

#%%
opt = Optimizer(f, 'PGD_LineSearch', n_points=100, long=0.001, 
                proyection=proyection_to_the_circle, x0=x0)
name = 'PGD Line-Search Convexo'
plots_info(opt, name)


#%%
x0 = np.full((2,1), 1/2)
opt = Optimizer(f, 'PGD_DecreasingStep', constant=constant, 
                proyection=proyection_out_of_the_circle, x0=x0)
name = 'PGD Paso Decreciente No-Convexo x0=(0.5,0.5)'
plots_info(opt, name)


#%%
x0 = np.full((2,1), 1/2)
opt = Optimizer(f, 'PGD_LineSearch', n_points=100, long=0.01, 
                proyection=proyection_out_of_the_circle, x0=x0)
name = 'PGD Line-Search No-Convexo x0=(0.5,0.5)'
plots_info(opt, name)