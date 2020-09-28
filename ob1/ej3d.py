#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 01:42:31 2020

@author: camilo
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 15
#%%

# Ploteo de curvas de nivel de f: R2-->R
fx = -0.1
fy = -0.1
f = lambda x,y: np.linalg.norm(np.array([x,y])-np.array([2,1/2]))**2+1
f_ = '$||x-x^*||^2+1$'

x = np.linspace(-0.5, 2.5, 100)
y = np.linspace(-1., 2., 100)

xx, yy = np.meshgrid(x, y)

z = np.zeros_like(xx)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i,j] = f(xx[i,j], yy[i,j])

z[np.isnan(z)] = 1000


levels = np.sort(np.append(np.linspace(-10, 10, 50),2))

plt.figure(figsize=(10,10))
plt.title(f'Curvas de nivel de f = {f_}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
contours = plt.contour(xx, yy, z, levels, linewidths=2, color='black')
plt.clabel(contours, inline=True, fontsize=8)



#%%

# Ploteo de region a partir de restricciones
# No me gusta mucho el plot

r1 = lambda x,y:x<1
r2 = lambda x,y:0<x
r3 = lambda x,y:y<1
r4 = lambda x,y:0<y

r = [r1, r2, r3, r4]


z = True
for rs in r:
    z &= rs(xx,yy)

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.imshow(z, cmap='Greys', origin='lower',
           extent=(xx.min(),xx.max(),yy.min(),yy.max()), alpha=0.7)
