import numpy as np

class FunctionToMinimize():
    def __init__(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
    def __call__(self, x):
        return self.forward(x)


class MinSquareRoot(FunctionToMinimize):    
    def __init__(self, A, b):
        self.A = A
        self.b = b
    
    def forward(self, x):
        self._x_is_valid_shape(x)
        return np.linalg.norm(self.A@x-self.b)**2
    
    def backward(self, x):
        self._x_is_valid_shape(x)
        return 2*self.A.T@(self.A@x-self.b)
    
    def min(self):
        return np.linalg.inv((self.A.T@self.A))@self.A.T@self.b
    
    def _x_is_valid_shape(self, x) -> None:
        if x.shape != (self.A.shape[1], 1):
            raise AttributeError("x debe se un vector columna")
            
class CustomFunction(FunctionToMinimize):
    def __init__(self, f, grad_f):
        self.f = f 
        self.grad_f = grad_f 
    def forward(self, x):
        self._x_is_valid_shape(x)
        x_ = x[0,0]
        y_ = x[1,0]
        return self.f(x_, y_)
    
    def backward(self, x):
        self._x_is_valid_shape(x)
        x_ = x[0,0]
        y_ = x[1,0]
        return np.array(self.grad_f(x_, y_)).reshape((2,1))
    
    def _x_is_valid_shape(self, x) -> None:
        if x.shape != (2, 1):
            raise AttributeError("x debe se un vector [2x1]")

            
    
    
            
         