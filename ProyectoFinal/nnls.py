import numpy as np
from time import time
def nnls(Z:np.ndarray, 
         X:np.ndarray, 
         tolerance: int = 1e-5,
         verbose: bool = True) -> np.ndarray:
    '''
    Resuelve el problema:
        min_d \\x - Zd\\^2 
        s.t d>0
    Codigo basado en Algorithmn NNLS de http://xrm.phys.northwestern.edu/research/pdf_papers/1997/bro_chemometrics_1997.pdf
    Parameters
    ----------
    Z : np.ndarray
        Diccionario de vectores. 
    X : np.ndarray
        Entradas que se quiere proyectar sobre D.
    verbose : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    d : TYPE
        Se retorna la matriz que indica la proyeccion. Es una matriz columna

    '''
    X = X.reshape((X.shape[0], 1))
    ti = time()
    # Checkeos de integridad
    if (Z<0).any():
        raise AttributeError('Alguna componente de Z es negativo')
    if (X<0).any():
        raise AttributeError('Alguna componente de X es negativo')
    if X.shape[1] !=1:
        raise AttributeError('X debe ser una matriz columna')
    if X.shape[0] != Z.shape[0]:
        raise AttributeError('X y Z no tienen tamaños compatibles')
    # Calculo de tamano de d
    M = Z.shape[1]
    # A1
    P = set()
    # A2
    R = {i for i in range(M)}
    # A3
    d = np.zeros((M, 1))
    # A4: tiene una optimizacion para calcular menos veces un termino
    # mirar esta optimizacion en https://angms.science/doc/NMF/nnls_pgd.pdf
    Q = Z.T@Z
    p = Z.T@X
    w_expr = lambda d:-2*(Q@d-p)
    w = w_expr(d)
    
    i = 0
    while True:
        i += 1
        print(i)
        if P: 
            w[np.array(list(P))] = -np.inf
        # B1
        if R and w.max()>tolerance:
            #B2
            index = w.argmax()
            # B3
            R.remove(index)
            P.add(index)
            # B4
            Z_aux = Z[:,np.array(list(P))]
            Z_auxT = Z_aux.T
            s = np.linalg.inv(Z_auxT@Z_aux)@Z_auxT@X
            if s.min()<0:
                #print(f'caso raro {i}')
                dn = d[np.array(list(P))]
                sn = s
                # C2
                alpha = -np.min(dn / (dn-sn))
                # C3
                d[np.array(list(P))] = d[np.array(list(P))] *(1-alpha) + alpha * s
                # C4
                R = set(np.where(d==0)[0])
                P = {i for i in range(M)} - R
                # C5
                Z_aux = Z[:,np.array(list(P))]
                Z_auxT = Z_aux.T
                s = np.linalg.inv(Z_auxT@Z_aux)@Z_auxT@X
            # B5
            d[np.array(list(P))] = s
            # B6
            w = w_expr(d)
        else:
            break
    cost = np.linalg.norm(X - Z@d, ord=2)**2
    if verbose: print(f'Cantidad de iteraciones: {i}\nTiempo: {time()-ti:.2f}')
    return d.reshape((d.shape[0],)), cost
    