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

xss = dict()
fss = dict()
ks = dict()
for opt in [PGD, ADMM]:
    t = 0
    N = 1000
    for _ in range(N):
        f = minimosCuadrados(A, b)
        g = regularizacionL1(l)
        
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
    print(f'{opt.__name__}:\nx_opt = {x} \niteraciones = {stop_condition.total_iter()}\ntime = {(t)*1000:.3f}ms')
    
    xss[opt.__name__] = np.array(xs)
    fss[opt.__name__] = np.array(fs)
    ks[opt.__name__] = np.arange(len(xs))
    
    #%%

plt.figure(figsize=(20,12))
for opt in [PGD, ADMM]:
    xs = xss[opt.__name__]
    fs = fss[opt.__name__]
    k = ks[opt.__name__]
    plt.plot(k, fs, label=f'{opt.__name__}')
    plt.ylim((0.9*np.min(fs),16))
    plt.ylabel(r'$f(x)+g(x)$')
    plt.xlabel(r'$k$')
    plt.grid(True)
plt.title('Valor de la función objetivo con las iteraciones')
plt.legend()
plt.savefig(f'figures/f_k.png')