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
from Error import ErrorToOptim, Error
from utils import calc_trajectory, plot_trayectory, plot_error, print_info


f_expr = lambda x,y: 5*x**2 + 5*y**2 + 5*x - 3*y - 6*x*y +5/4
grad_f = lambda x,y: (10*x + 5 - 6*y,  10*y - 3 - 6*x) 

f = CustomFunction(f_expr, grad_f)
error_to_optim = ErrorToOptim(np.array([[-0.5],[0]]))
error = Error()

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
plt.close('all')
opt = Optimizer(f, 'PGD', constant=constant, 
                proyection=proyection_to_the_circle, x0=x0)

k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria PGD Convexo', k=k)
plt.figure()
plot_error(e, title='PGD Convexo')
print_info(k,x,e,error,'PGD Convexo')


#%%
plt.close('all')
x0 = np.full((2,1), 1/2)
opt = Optimizer(f, 'PGD', constant=constant, 
                proyection=proyection_out_of_the_circle, x0=x0)

k, x, e = calc_trajectory(opt, error, error_to_optim)
plt.figure()
plot_trayectory(x, title='Trayectoria PGD No Convexo', k=k)
plt.figure()
plot_error(e, title='PGD No Convexo')
print_info(k,x,e,error,'PGD No Convexo')