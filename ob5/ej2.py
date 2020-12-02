# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(5)
import matplotlib.pyplot as plt

gatos = np.loadtxt('Gatos.asc').astype(np.uint8)
conejos = np.loadtxt('Conejos.asc').astype(np.uint8)

#%%
def vect2img(x):
    return np.transpose((x[:-1].reshape(3, 256,256)), (2,1,0))


plt.figure()
plt.imshow(vect2img(gatos[:, 0]))
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
    
def F(x, y, a, n):
    if a.T@x <= 0:
        value = epsilon
        #print('chico'+ ' '+ n)
    else:
        value = 1
        #print('grande'+ ' '+ n)
    return 2 * (-y + ReLU(a.T@x, epsilon)) * value * x

# gatos   : label=0
# conejos : label=1
gato_label = 0
conejo_label = 1

step = ConstantStep(1e-9)

a = np.zeros((gatos.shape[0],1))
ases = list()
ases.append(a)
#%%
from tqdm import tqdm
K = 20
N_EPOCHS = 100
for n in tqdm(range(N_EPOCHS)):
    for k in range(K):
        gato = gatos[:, k:k+1]
        conejo = conejos[:, k:k+1]
        a = a - step.step()*F(gato, gato_label, a, 'gato')
        #print(np.unique(a))
        #ases.append(a)
        a = a - step.step()*F(conejo, conejo_label, a, 'conejo')
        #print(np.unique(a))
        #ases.append(a)
        
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
    
#%%
plt.figure(figsize=(7,7))
for k in range(20):
    plt.scatter(gatos[:,k]@a, k, c='g', label='gatos')
    plt.scatter(conejos[:,k]@a, k, c='r', label='gatos')
plt.axvline(0.5)
plt.legend(['linea divisoria', 'gatos', 'conejos'])
plt.title('Ejemplos de train')
plt.xlabel(r'$a^Tx$')
plt.ylabel('Indice')
plt.grid(True)
plt.savefig('figures/ej2_train')

#%%
plt.figure(figsize=(7,7))
for k in range(20,30):
    plt.scatter(gatos[:,k]@a, k, c='g', label='gatos', marker='o')
    plt.scatter(conejos[:,k]@a, k, c='r', label='gatos', marker='x')
plt.axvline(0.5)
plt.legend(['linea divisoria', 'gatos', 'conejos'])
plt.title('Ejemplos de validación')
plt.xlabel(r'$a^Tx$')
plt.ylabel('Indice')
plt.grid(True)
plt.savefig('figures/ej2_val')

#%%
def a2img(a):
    v = vect2img(a)
    v -= v.min()
    v /= v.max()
    v *= 255
    v = v.astype(np.uint8)
    return v


import imageio
plt.figure()
plt.imshow(v:=a2img(a))
#plt.title(fr'$a$')
plt.axis(False)
imageio.imsave('figures/ej2_a.png', v)


