
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 15
#%%

# Ploteo de curvas de nivel de f: R2-->R
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

plt.figure(figsize=(10,10))
plt.title(f'Curvas de nivel de $f(x,y) = {f_}$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
contours = plt.contour(xx, yy, z, levels, linewidths=2, color='black')
plt.clabel(contours, inline=True, fontsize=8)



#%%

# Ploteo de region a partir de restricciones
# No me gusta mucho el plot

r1 = lambda x,y:x**2+y**2<=(1/4)**2

r = [r1]


z = True
for rs in r:
    z &= rs(xx,yy)

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.imshow(z, cmap='Greys', origin='lower',
           extent=(xx.min(),xx.max(),yy.min(),yy.max()), alpha=0.7)
