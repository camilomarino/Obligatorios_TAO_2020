import numpy as np

class Function:
    def __init__(self):
        raise NotImplementedError
    def __call__(self, x):
        return self.forward(x)
        
    
class minimosCuadrados(Function):
    '''
     f(x) = 1/2 * ||Ax-b||**2
    '''
    def __init__(self, A, b):
        self.A = A
        self.b = b
    def forward(self, x):
        return 1/2 * np.linalg.norm(self.A@x-self.b, ord=2)**2
    def grad(self, x):
        '''
        grad = At @ (A@x - b)
        '''
        return self.A.T @ (self.A@x - self.b)
    
    
class regularizacionL1(Function):
    '''
    f = lambda*||x|| (norma 1)
    '''
    def __init__(self, l):
        self.l = l
    def forward(self, x):
        return self.l * np.linalg.norm(x, ord=1)
    def grad(self, x):
        return self.l * np.sign(x)
    def proximal(self, x, alpha):
        prox = np.zeros_like(x)
        prox[x/alpha>1], prox[x/alpha<-1] = (x[x/alpha>1]-alpha, 
                                            x[x/alpha<-1]+alpha)
        return prox
   
    
class Optimizer:
    def __init__(self):
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
        
        
class PGD(Optimizer):
    '''
    Proximal gradient descent
    '''
    def __init__(self, f, g, alpha):
        self.f = f
        self.g = g
        self.alpha = alpha
        
    def step(self, x):
        '''
        Dado un xk retorna el xk+1
        '''
        # import ipdb; ipdb.set_trace()
        v = x - self.alpha*self.f.grad(x)
        x_siguiente = self.g.proximal(v, self.alpha)
        return x_siguiente


class ThresholdDiffStop:
    def __init__(self, diff):
        self.diff = diff
        self.iter = 0
    def __call__(self, x_nuevo) -> bool:
        if self.iter==0:
            self.x_ant = x_nuevo
            self.iter += 1
            return False
        condition = np.linalg.norm(x_nuevo-self.x_ant)<self.diff
        self.x_ant = x_nuevo
        self.iter += 1
        return condition
    def total_iter(self):
        return self.iter
    
