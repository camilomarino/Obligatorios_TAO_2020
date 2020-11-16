import numpy as np
import matplotlib.pyplot as plt
from time import time
from utils import minimosCuadrados, regularizacionL1, PGD, ADMM, ThresholdDiffStop

plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['font.size'] = 24

A = np.loadtxt('data/A.asc')
b = np.loadtxt('data/b.asc')
alpha = 1/np.linalg.norm(A.T@A, ord=2)
l = 0.15

f = minimosCuadrados(A, b)
g = regularizacionL1(l)

#%%
xss = dict()
fss = dict()
ks = dict()

opimizadores = [PGD, ADMM, ADMM]
alphas = [alpha, alpha, 10*alpha]
for opt, alpha in zip(opimizadores, alphas):
    t = 0
    N = 10000
    for _ in range(N):
        
        optimizer = opt(f, g, alpha)
        
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
        t += (t2-t1)/N
    print(f'{opt.__name__} (alpha={alpha*1000:.3f}e-3):\nx_opt = {x} \niteraciones = {stop_condition.total_iter()}\ntime = {(t)*1000:.3f}ms')
    print()
    xss[opt.__name__+str(alpha)] = np.array(xs)
    fss[opt.__name__+str(alpha)] = np.array(fs)
    ks[opt.__name__+str(alpha)] = np.arange(len(xs))
    
    #%%

plt.figure(figsize=(20,12))
for j, (opt, alpha) in enumerate(zip(opimizadores, alphas)):
    if j==0: i=1
    if j==1: i=1
    if j==2: i=2
    xs = xss[opt.__name__+str(alpha)]
    fs = fss[opt.__name__+str(alpha)]
    k = ks[opt.__name__+str(alpha)]
    plt.plot(k, fs, label=fr'{opt.__name__} $\alpha_{i}$')
    plt.ylim((0.9*np.min(fs),16))
    plt.ylabel(r'$f(x)+g(x)$')
    plt.xlabel(r'$k$')
    plt.grid(True)
plt.title('Valor de la función objetivo con las iteraciones')
plt.legend()
plt.savefig(f'figures/f_k.png')