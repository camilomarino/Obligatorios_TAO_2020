import numpy as np

class ErrorToOptim:
    def __init__(self, x_sol):
        self.x_sol = x_sol
    def __call__(self, xk):
        return np.linalg.norm(self.x_sol-xk)/np.linalg.norm(self.x_sol)
    

class Error:
    def __init__(self):
        pass
    def __call__(self, xk, xk_1):
        return np.linalg.norm(xk-xk_1)/np.linalg.norm(xk_1)
    
    
class ErrorToPreviousSample:
    def __init__(self):
        pass
    def __call__(self, xk, xk_1):
        return np.linalg.norm(xk-xk_1)