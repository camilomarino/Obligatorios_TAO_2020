#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:25:57 2020

@author: camilo
"""

def level_curves(f, x_domain, y_domain, levels=None):
    if isinstance(x_domain, tuple):
        x_domain = np.linspace(x_domain[0], x_domain[1], 1000)
    
    if isinstance(y_domain, tuple):
        y_domain = np.linspace(y_domain[0], y_domain[1], 1000)
        
    xx, yy = np.meshgrid(x_domain, y_domain)
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p = np.array([[xx[i,j]],[yy[i,j]]])
            zz[i,j] = f(p)
    if levels is None:
        plt.contour(xx, yy, zz)
    else:
        plt.contour(xx, yy, zz, levels)

level_curves(f, (-1000,1000), (-1000,1000))
    
#%%
n=12
x = x[:,:n]
k = k[:n]
x_min, y_min = x.min(axis=1)
x_max, y_max = x.max(axis=1)
# levels = list()
# for i in range(n):
#     levels.append(f(x[:,i:i+1]))
# levels.sort()
# levels = np.array(levels)
levels = np.linspace(0, int(1e7), int(1e5))
level_curves(f, (x_min, x_max), (y_min, y_max))

#%%
plt.scatter(x[0], x[1])
for i in range(k.shape[0]):
    plt.annotate(k[i], (x[0,i], x[1,i]))