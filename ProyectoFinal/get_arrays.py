import numpy as np

def get_Q(mask):
    Q = np.zeros( (len(np.unique(mask)), len(mask)) )
    for i, num_elec in enumerate(np.unique(mask)):
        Q[i, mask==num_elec] = 1
    return Q
    
    

#%%
def get_U(mask, T):
    U = np.ones( (len(np.unique(mask)), T) )
    return U

def get_G(X, U, beta):
    return np.vstack((X, beta*U))

def get_H(D, Q, beta):
    return np.vstack((D, beta*Q))