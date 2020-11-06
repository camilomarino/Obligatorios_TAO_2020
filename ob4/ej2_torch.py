import numpy as np
import torch
device = 'cuda'

#%%
A = np.loadtxt('data/A.asc')
b = np.loadtxt('data/b.asc')

alpha = 1/np.linalg.norm(A.T@A, ord=2)

A = torch.tensor(A, device=device)
b = torch.tensor(b, device=device).view((-1,1))

#%%
x = torch.zeros(size=(2,1), dtype=torch.float64, requires_grad=True, device=device)

#%%
l = 0.15
def lossLASSO(x, A, b):
    return (1/2  * torch.linalg.norm(A@x -b, ord=2)**2 + 
            l*torch.linalg.norm(x, ord=1))

#%%

from torch import optim

optimizer = optim.SGD([x], lr=alpha)

#%%
for i in range(100):
    optimizer.zero_grad()
    loss = lossLASSO(x, A, b)
    loss.backward()
    optimizer.step()
    
print('x = ', x)
print('loss = ', lossLASSO(x, A, b))