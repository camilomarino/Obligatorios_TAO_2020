#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:08:26 2020

@author: camilo
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Plot de una funcion en una region
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 12

plt.rcParams['figure.figsize'] = [16, 10]

plt.close('all')
def plot_f(f, f_nombre, label='f(x)', new_figure=True, vmin=-1.2, vmax=1.2):
    x = np.linspace(vmin, vmax, 1000)
    y  = f(x)
    
    #
    if new_figure:
        plt.figure()
        plt.axvline(1, c='blue', alpha=0.4, label='limites')
        plt.axvline(-1, c='blue', alpha=0.4)
        plt.axvline(0, c='black')
        plt.axhline(0, c='black')
    plt.plot(x, y, label=label)
    plt.legend(loc='upper right')
    plt.title(f'f(x) = {f_nombre}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    

f1 = lambda x:4*x**4-x**3-4*x**2+1, '$4x^4-x^3-4*x^2+1$'
f2 = lambda x:x**3, '$x^3$'
f3 = lambda a:(lambda x:(x-a)**2+1, f'$(x-1)^2+{a}$')
#%%
plot_f(*f1)
plot_f(*f2)
#%%
new_figure = True
for a in [-2,-1,0,1,2]:
    plot_f(*f3(a), new_figure=new_figure, label=f'a = {a}', vmin=-4, vmax=4)
    new_figure = False


