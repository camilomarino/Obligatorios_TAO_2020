# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)


class ConstantStep:
    def __init__(self, const):
        self.const = const
    def step(self, *k1, **k2):
        return self.const
    def __str__(self):
        return fr'$\alpha_k = {self.const}$'
class DecreasingStep:
    def __init__(self, const, *k1, **k2):
        self.const = const
    def step(self, k):
        return self.const/k
    def __str__(self):
        return fr'$\alpha_k = {self.const}/k$'

A = np.array([[2, 1],
              [1, 2]])
theta_0 = np.array([[1],
                    [1]])


X = lambda :A + np.random.randn(2,2)
y = lambda X: X@theta_0 + np.random.randn(2,1)

Ff = lambda X, y, theta: 2*X.T @ (X@theta - y)

stepsTypes = [ConstantStep(0.1), ConstantStep(0.01), ConstantStep(0.001),
         DecreasingStep(0.1)]


plt.figure(figsize=(7,7))
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Trayectoria $\theta_k$')

for step_type in stepsTypes:
    thetas = list()
    theta_k = np.zeros((2,1))
    thetas.append(theta_k)
    
    alpha_const = 0.1
    
    K = 10000
    for k in range(K):
        X_ = X()
        y_ = y(X_)
        F = Ff(X_, y_, theta_k)
        alpha_k = step_type.step(k+1)
        theta_k = theta_k - alpha_k * F
        thetas.append(theta_k)
        
    print(theta_k)
    
    thetas_ = np.array(thetas)[:,:,0]
    
    plt.plot(thetas_[:, 0], thetas_[:, 1], label=str(step_type))
    
    plt.scatter(1, 1, c='r')
plt.xlim(-.25,2.25)
plt.ylim(-.25,2.25)
plt.grid(True)

plt.legend()
plt.savefig('figures/ej1.png')