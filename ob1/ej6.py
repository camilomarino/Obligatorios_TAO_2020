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


f_expr = lambda x,y: 5*x**2 + 5*y**2 + 5*x - 3*y - 6*x*y +5/4
grad_f = lambda x,y: (10*x + 5 - 6*y,  10*y - 3 - 6*x) 

f = CustomFunction(f_expr, grad_f)
error_to_optim = ErrorToOptim(np.array([[-0.5],[0]]))
error = ErrorToPreviousSample()

x0 = np.full((2,1), 0)

constant = 1

def proyection_to_the_circle(x, R=0.5):
    norm = np.linalg.norm(x)
    if norm>1.0:
        x = x/norm
    return x
       
def proyection_out_of_the_circle(x, R=0.5):
    norm = np.linalg.norm(x)
    if norm<1.0:
        x = x/norm
    return x 
    
#%%
def plots_info(opt, name):
    plt.close('all')
    
    k, x, e, fxk = calc_trajectory(opt, error, error_to_optim, more_data=True)

    plt.figure()
    plot_trayectory(x, title=f'Trayectoria {name}', k=k)
    plt.figure()
    plot_error(e, title=f'{name}')
    print_info(k,x,e,error,f'{name}')
    plot_f_evolution(fxk, title=f'Evolucion de f {name}')

#%%
opt = Optimizer(f, 'PGD_DecreasingStep', constant=constant, 
                    proyection=proyection_to_the_circle, x0=x0)
name = 'PGD Paso Decreciente Convexo'
plots_info(opt, name)


#%%
x0 = np.full((2,1), 1/2)
opt = Optimizer(f, 'PGD_DecreasingStep', constant=constant, 
                proyection=proyection_out_of_the_circle, x0=x0)
name = 'PGD Paso Decreciente No-Convexo'
plots_info(opt, name)

#%%
opt = Optimizer(f, 'PGD_LineSearch', n_points=100, long=0.001, 
                proyection=proyection_to_the_circle, x0=x0)
name = 'PGD Line-Search Convexo'
plots_info(opt, name)


#%%
opt = Optimizer(f, 'PGD_LineSearch', n_points=100, long=0.001, 
                proyection=proyection_out_of_the_circle, x0=x0)
name = 'PGD Line-Search No-Convexo'
plots_info(opt, name)