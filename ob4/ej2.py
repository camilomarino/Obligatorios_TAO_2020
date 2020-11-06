import numpy as np
import matplotlib.pyplot as plt
from time import time
from utils import minimosCuadrados, regularizacionL1, PGD, ThresholdDiffStop

plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['font.size'] = 24

A = np.loadtxt('data/A.asc')
b = np.loadtxt('data/b.asc')
alpha = 1/np.linalg.norm(A.T@A, ord=2)
l = 0.15


f = minimosCuadrados(A, b)
g = regularizacionL1(l)

optimizer = PGD(f, g, alpha)

stop_condition = ThresholdDiffStop(0.0001)

x = np.zeros(2)
xs = [x]
fs = [f(x)+g(x)]

t1 = time()

while not stop_condition(f(x)+g(x)):
    x = optimizer.step(x)
    xs.append(x)
    fs.append(f(x)+g(x))
t2 = time()

print(f'x_opt = {x} \niteraciones = {stop_condition.total_iter()}\ntime = {(t2-t1)*1000:.3f}ms')

xs = np.array(xs)
fs = np.array(fs)
k = np.arange(len(xs))

plt.figure(figsize=(20,12))
plt.plot(k, fs)
plt.ylim((0.9*np.min(fs),1.1*fs[1]))
plt.ylabel(r'$f(x)+g(x)$')
plt.xlabel(r'$k$')
plt.title('Valor de la función objetivo con las iteraciones')
plt.grid(True)