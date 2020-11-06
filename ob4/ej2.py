import numpy as np
from time import time
from utils import minimosCuadrados, regularizacionL1, PGD, ThresholdDiffStop


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


t1 = time()

while not stop_condition(x):
    x = optimizer.step(x)
    xs.append(x)
t2 = time()

print(f'x_opt = {x} \niteraciones = {stop_condition.total_iter()}\ntime = {t2-t1:.4f}s')