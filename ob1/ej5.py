#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:59:31 2020

@author: camilo
"""

import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)

A = np.array([[-4.100000000000000000e+01, 2.000000000000000000e+01],
            [-4.600000000000000000e+01, -8.000000000000000000e+00],
            [-5.000000000000000000e+00, -3.300000000000000000e+01],
            [-5.500000000000000000e+01, 1.000000000000000000e+00],
            [-5.500000000000000000e+01, -6.000000000000000000e+00]])

b = np.array([8.000000000000000000e+00 , 5.000000000000000000e+00, 
            -3.000000000000000000e+00, 1.000000000000000000e+01, 
            4.000000000000000000e+00])
b = b[:, np.newaxis]





#%%

# Parte a
x_opt = np.linalg.inv(A.T@A)@A.T@b

#%%
# Parte b
norm2_A = np.sqrt(np.max(np.linalg.eigvals(A.T@A)))
alpha = 0.0002#1/(2*norm2_A)

x = np.random.normal(size=(2,1))

xs = [x.copy()]
#%%
#import ipdb; ipdb.set_trace()
for k in range(10000):
    grad = 2*(A@x-b).T@A
    alpha = 0.001/(k+1)
    x -= alpha*grad.T
    if np.sum(np.isnan(x)) > 0:
        break
    xs.append(x.copy())


error_relativo = [np.linalg.norm(x_opt-x)/np.linalg.norm(x_opt) for x in xs]
plt.close('all')
plt.figure()
plt.plot(error_relativo)


#%%

plt.figure()

def f(x, y):
    X = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=2)[..., np.newaxis]
    return np.linalg.norm(A@X-b, axis=2)[...,0]**2

x = np.linspace(-3000, 3000, 1000)
y = np.linspace(-3000, 3000, 1000)

xx, yy = np.meshgrid(x, y)

z = f(xx, yy)


levels = np.linspace(1,50,100)

plt.grid(True)
plt.contour(xx, yy, z)

#%%

xs_ = np.array(xs)[:,:,0]
#xs = np.nan_to_num(xs, nan=77)
plt.scatter(xs_[:,0], xs_[:,1])
plt.scatter(x_opt[0,:], x_opt[1,:], c='red')