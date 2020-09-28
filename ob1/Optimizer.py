import numpy as np

dtype = np.float64
class Optimizer:
    '''
    type_step:
        constant: kwargs:
                    step
        
        decreasing: kwargs:
                    constant
        
        line_search: kwargs:
                        n_points: numero de puntos de la grilla
                        long: largo de la grilla
        armijo: kwargs:
                    sigma
                    beta
                    s
                    
    D: Matriz D que premultiplica al gradiente
    
    '''
    def __init__(self, f, type_step, **kwargs):
        self.f = f
        self.type = type_step
        self.step_driver = self._get_step_instance(f, type_step, **kwargs)
        self.k = 0
        self.xk = self._get_x0(x0=kwargs.get('x0'))
        if kwargs.get('D') is not None:
            self.D = kwargs.get('D')
        else :
            self.D = np.eye(2)
        
    def get_xk(self):
        return self.xk
    
    def get_k(self):
        return self.k
    
    def info(self):
        return {'k':self.k, 'xk':self.xk, 'type':self.type}
    
    def step(self):
        self.k += 1
        self.xk = self.step_driver.next_xk(xk=self.xk, k=self.k, D=self.D)
        return self.k, self.xk
    
    def _get_step_instance(self, f, type_step, **kwargs):
        if type_step=='constant' or type_step=='constante':
            return ConstantStep(f, **kwargs)
        elif type_step=='decreasing' or type_step=='decreciente':
            return DecreasingStep(f, **kwargs)
        elif type_step=='line_search':
            return LineSearch(f, **kwargs)
        elif type_step=='armijo':
            return Armijo(f, **kwargs)
        elif type_step=='PGD':
            return PGD(f, **kwargs)            
        else:
            raise AttributeError
  
    def _get_x0(self, x0=None):
        if x0 is None:
            return np.zeros((2,1), dtype=dtype)
        else:
            return x0
        
        
        
    
class Step():
    def __init__(self):
        raise NotImplementedError
    def next_step():
        raise NotImplementedError
        
class ConstantStep(Step):
    def __init__(self, f, step: float = 0.01, **kwargs):
        self.step = step
        self.f = f
        
    def next_xk(self, xk, k, D):
        xk = xk - self.step * D@self.f.backward(xk)
        return xk
   
    
class DecreasingStep(Step):
    def __init__(self, f, constant: float = 0.01, **kwargs):
        self.constant = constant
        self.f = f
        
    def next_xk(self, xk, k, D):
        step = self.constant/k
        xk = xk - step * D@self.f.backward(xk)
        return xk
        
        
class LineSearch(Step):
    def __init__(self, f, n_points=1000, long=1, **kwargs):
        self.n_points = n_points
        self.long = long
        self.f = f
        self.steps = self._get_steps()
        
    def next_xk(self, xk, k, D):
        w = D@self.f.backward(xk)
        xk_candidatos = np.tile(xk, self.steps.size) - self.steps * w
        #import ipdb; ipdb.set_trace()
        fxk_candidatos = list()
        for k in range(xk_candidatos.shape[1]):
            fxk_candidatos.append(self.f.forward(xk_candidatos[:,k:k+1]))
        fxk_candidatos = np.array(fxk_candidatos)
        
        arg_min = fxk_candidatos.argmin()
        return xk_candidatos[:, arg_min:arg_min+1]
    
    def _get_steps(self):
        return np.linspace(0, self.long, self.n_points)
    
    
class Armijo(Step):
    def __init__(self, f, sigma=0.1, beta=0.5, s=0.1, **kwargs):
        self.f = f
        self.sigma = sigma
        self.beta = beta
        self.s = s
    
    def next_xk(self, xk, k, D):
        #import ipdb; ipdb.set_trace()
        grad_f = self.f.backward(xk)
        dk = -D@grad_f
        alfa = self.s
        while True:            
            desigualdad_izq = (self.f(xk) -
                               self.f(xk+alfa*self.s*dk))
            desigualdad_der = -self.sigma*alfa*self.s*grad_f.T@dk
            if desigualdad_izq >= desigualdad_der:
                break
            alfa *= self.beta
        xk = xk+alfa*self.s*dk
        return xk
    
class PGD(Step):
    '''
    Caso particular de alpha = 1. Para otro caso el codigo seria más dificil,
    pero más general.
    Paso decreciente con el valor de constant
    '''
    def __init__(self, f, proyection, constant, **kwargs):
        self.f = f
        self.proyection = proyection
        self.constant = constant
    
    def next_xk(self, xk, k, D):
        step = self.constant/k
        xk = xk - step * D@self.f.backward(xk)
        xk = self.proyection(xk)
        return xk        
        
        