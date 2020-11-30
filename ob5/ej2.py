# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

gatos = np.loadtxt('Gatos.asc').astype(np.uint8)
conejos = np.loadtxt('Conejos.asc').astype(np.uint8)

#%%
def vect2img(x):
    return np.transpose((x[:-1].reshape(3, 256,256)), (2,1,0))


plt.figure()
plt.imshow(vect2img(gatos[:, 1]))
#%%

plt.figure()
plt.imshow(np.transpose((conejos[:-1, 10].reshape(3, 256,256)), (2,1,0)))
#%%

class ConstantStep:
    def __init__(self, const):
        self.const = const
    def step(self, *k1, **k2):
        return self.const
    def __str__(self):
        return fr'Paso constante $\alpha_k = {self.const}$'
    
epsilon = 0.1

def ReLU(x, epsilon):
    if x>0:
        return x
    else:
        return epsilon * x
    
def F(x, y, a):
    if a.T@x <= 0:
        value = epsilon
    else:
        value = 1
    return 2 * (-y + ReLU(a.T@x, epsilon)) * value * x

# gatos   : label=0
# conejos : label=1
gato_label = 0
conejo_label = 1

step = ConstantStep(1e-9)

a = np.zeros((gatos.shape[0],1))

#%%
K = 20
N_EPOCHS = 50
for n in range(N_EPOCHS):
    for k in range(K):
        gato = gatos[:, k:k+1]
        conejo = conejos[:, k:k+1]
        
        a = a - step.step()*F(gato, gato_label, a)
        a = a - step.step()*F(conejo, conejo_label, a)
        
#%%
# Train
label = lambda x: 'gato' if x<0.5 else 'conejo'

print('GATOS')
for k in range(0,10):
    gato = gatos[:, k:k+1]
    v = a.T@gato
    print(v, label(v))

print('\n\nCONEJOS')
for k in range(0,10):
    conejo = conejos[:, k:k+1]
    v = a.T@conejo
    print(v, label(v))
#%%
# Test
label = lambda x: 'gato' if x<0.5 else 'conejo'

print('GATOS')
for k in range(20,30):
    gato = gatos[:, k:k+1]
    v = a.T@gato
    print(v, label(v))

print('\n\nCONEJOS')
for k in range(20,30):
    conejo = conejos[:, k:k+1]
    v = a.T@conejo
    print(v, label(v))