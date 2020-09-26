#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:51:04 2020

@author: camilo
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

# Ploteo de curvas de nivel de f: R2-->R
fx = -0.1
fy = -0.1
f = lambda x,y: np.sin(2*np.pi*fx*x)*np.sin(2*np.pi*fy*y)#(-x**2+y**2)
f_ = '$-log(y^2 - x^2)$'

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

xx, yy = np.meshgrid(x, y)

z = f(xx, yy)

z[np.isnan(z)] = 1000


levels = np.linspace(-1,1,10)

# plt.figure()
# plt.title(f'f = {f_}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)
# plt.contour(xx, yy, z, levels)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')


# Plot the surface.
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#%%

# Ploteo de region a partir de restricciones
# No me gusta mucho el plot

r1 = lambda x,y:x**2+y**2<=1
r2 = lambda x,y:2*x-y<=0
r3 = lambda x,y:y>=1/2
r4 = lambda x,y:x>=0

r = [r1, r2, r3, r4]


z = True
for rs in r:
    z &= rs(xx,yy)

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.imshow(z, cmap='Greys', origin='lower',
           extent=(xx.min(),xx.max(),yy.min(),yy.max()), alpha=0.7)
