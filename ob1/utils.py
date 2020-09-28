import matplotlib.pyplot as plt
import numpy as np

path_base = 'plots'

def calc_trajectory(opt, error, error_to_optim, iteraciones=1000, 
                    min_error=0.0001, more_data=False):
    k, x, e = [opt.k], [opt.xk], [error_to_optim(opt.xk)]
    if more_data: f = [opt.f(opt.xk)]
    for iter in range(1,iteraciones):
        i, xk = opt.step()
        k.append(i)
        x.append(xk)
        e.append(error_to_optim(opt.xk))
        # if iter>3:
        #     print(error(x[-1], x[-2]))
        if more_data: f.append(opt.f(xk))
        if iter>3 and error(x[-1], x[-2])<min_error:
            break
        
        
        
        
    x = np.array(x)[...,0].T 
    k = np.array(k)
    if more_data: f=np.array(f)
    if more_data: return k,x,e,f
    return k, x, e
        
def plot_trayectory(x, title, annotate=True, k=None):
    plt.scatter(x[0], x[1])
    #plt.plot(x[0], x[1])
    if annotate: 
        for i in range(k.shape[0]):
            if i%7==0:
                plt.annotate(k[i], (x[0,i], x[1,i]))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.savefig(path_base+'/'+title+'.png')

def plot_error(e, title=None):
    plt.plot(e, label=title)
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    if title is not None: plt.title(title)
    plt.savefig(path_base+'/'+title+'.png')
    
def print_info(k,x,e, error, tipo=""):
    x_ = x[0,-1]
    y_ = x[1,-1]
    k_ = k[-1]
    e_ = e[-1]
    print('-'*55)
    print(tipo)
    print(f'El punto optimpo es ({x_:.5f},{y_:.5f})')
    print(f'Con {k_} iteraciones')
    print(f'Un error final de {e_:.5f}')
    print(f'Un error relativo respecto al paso anterior de {error(x[:,-1], x[:,-2])}')
    
def plot_f_evolution(fs, title):
    plt.figure()
    plt.plot(fs, label=title)
    plt.xlabel('Iteración')
    plt.ylabel('$f(x_k)$')
    if title is not None: plt.title(title)
    plt.savefig(path_base+'/'+title+'.png')
    
