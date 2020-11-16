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
        self.first_matrix = dict()
        self.second_matrix = dict()
    def forward(self, x):
        return 1/2 * np.linalg.norm(self.A@x-self.b, ord=2)**2
    def grad(self, x):
        '''
        grad = At @ (A@x - b)
        '''
        return self.A.T @ (self.A@x - self.b)
    def proximal(self, x, alpha):
        #Optimizacion para no invertir siempre la matriz
        if alpha in self.first_matrix.keys():
            first_matrix = self.first_matrix[alpha]
        else:
            first_matrix = np.eye(x.shape[0]) + alpha*self.A.T@self.A
            first_matrix = np.linalg.inv(first_matrix) 
            self.first_matrix[alpha] = first_matrix
        if alpha in self.second_matrix.keys():
            second_matrix = self.second_matrix[alpha]
        else:
            second_matrix = alpha*self.A.T@self.b
            self.second_matrix[alpha] = second_matrix
        return first_matrix @ (x + second_matrix)
    def min(self):
        return np.linalg.inv(self.A.T@self.A) @ self.A.T @ self.b
    
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
    __name___ = 'PGD'
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


class ADMM(Optimizer):
    '''
    A diferencia de PGD, tiene memoria, por lo que no alcanza solo con el
    x anterior. Internamente guardo z y u. No guardo x (lo recibo como
    parametro en step) para tener la misma interfaz que PGD
    '''
    __name__ = 'ADMM'
    def __init__(self, f, g, alpha):
        self.f = f
        self.g = g
        self.alpha = alpha
        self.u = None
        self.z = None
        
    def step(self, x):
        if self.u is None and self.z is None:
            self._init_uz(x.shape)
        
        x_siguiente = self.f.proximal(self.z - self.u, self.alpha)
        z_siguiente = self.g.proximal(x_siguiente + self.u, self.alpha)
        u_siguiente = self.u + x_siguiente - z_siguiente
        
        self.u = u_siguiente
        self.z = z_siguiente
        
        return x_siguiente
            
    def _init_uz(self, shape):
        self.u = np.zeros(shape)
        self.z = np.zeros(shape)

class ThresholdDiffStop:
    def __init__(self, diff):
        self.diff = diff
        self.iter = 0
    def __call__(self, x_nuevo) -> bool:
        if self.iter==0:
            self.x_ant = x_nuevo
            self.iter += 1
            return False
        condition = np.linalg.norm(x_nuevo-self.x_ant)<self.diff or self.iter>=100
        self.x_ant = x_nuevo
        self.iter += 1
        return condition
    def total_iter(self):
        return self.iter
    
