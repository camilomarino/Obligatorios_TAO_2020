#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:26:40 2020

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
def plot_f(f, f_nombre, label='f(a)', new_figure=True, vmin=-15, vmax=15):
    x = np.linspace(vmin, vmax, 1000)
    y  = f(x)
    
    #
    if new_figure:
        plt.figure()
        plt.axvline(0, c='black')
        plt.axhline(0, c='black')
    plt.plot(x, y, label=f'f(a) = {f_nombre}')
    plt.legend(loc='upper right')
    plt.title(f'f(a)')
    plt.xlabel('a')
    plt.ylabel('f(a)')
    plt.grid(True)
    

f1 = lambda a:a**2+2*a+2, '$a^2+2a+2$'
f2 = lambda a:a**2-2*a+2, '$a^2-2a+2$'
f3 = lambda a:np.ones_like(a), '1'

#%%
plot_f(*f1)
#%%
plot_f(*f2, new_figure=False)
#%%
plot_f(*f3, new_figure=False)