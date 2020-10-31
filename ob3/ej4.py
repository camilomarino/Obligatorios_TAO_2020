# Import packages.
import cvxpy as cp
import numpy as np




def solution_ej_4(R, d2, d3, verbose=False):
    p = cp.Variable(3)
    g = cp.Variable(2)
    t = cp.Variable()
    
    cost = g[0] + t
    
    constraints = [ [1,0,1]@p+[-1,0]@g==0, [0,1,1]@p+[0,1]@g==d2, [1,-1,0]@p==d3 , 
                   [1,1,-1]@p==0 , [0,1,0]@p<= R ,[0,-1,0]@p<= R, g[0]>=0, g[1]>=0,
                   t>=0, t>=4*(g[1]-40)]
    prob = cp.Problem(cp.Minimize(cost),constraints)
    prob.solve()
    
    opt = prob.value
    g1 = g.value[0]
    g2 = g.value[1]
    
    l1 = constraints[0].dual_value
    l2 = constraints[1].dual_value
    l3 = constraints[2].dual_value
    
    nu = constraints[3].dual_value
    
    mu1 = constraints[4].dual_value
    mu2 = constraints[5].dual_value
    mu3 = constraints[6].dual_value
    mu4 = constraints[7].dual_value
    mu5 = constraints[8].dual_value
    mu6 = constraints[9].dual_value
    
    p1 = p.value[0]
    p2 = p.value[1]
    p3 = p.value[2]
    
    t = t.value
    
    
    if verbose:
        print("El costo óptimo es", prob.value)
        print()
        print("La generación óptima es %s" %g.value)
        print(f"t = {t}")
        print()
        
        print("Los flujos por las líneas son %s" %p.value)
        print()
        
        print("lambda_1= %s" %constraints[0].dual_value)
        print("lambda_2= %s" %constraints[1].dual_value)
        print("lambda_3= %s" %constraints[2].dual_value)
        
        print()
        print("nu, %s" %constraints[3].dual_value)
        print()
        
        print("mu_{1}= %s" %constraints[4].dual_value)
        print("mu_{2}= %s" %constraints[5].dual_value)
        print("mu_{3}= %s" %constraints[6].dual_value)
        print("mu_{4}= %s" %constraints[7].dual_value)
        print("mu_{5}= %s" %constraints[8].dual_value)
        print("mu_{6}= %s" %constraints[9].dual_value)
        return
    return [opt, g1, g2, l1, l2, l3, nu, mu1, mu2, mu3, mu4, mu5, mu6, p1, p2, p3, t]
#            0    1   2   3   4   5   6   7    8    9    10   11   12  13  14  15
#%%
# Datos
R=30
d3=10
values = list()
#plt.figure()
rango = range(1, 205)
for d2 in rango:
    values.append(solution_ej_4(R, d2, d3))
   
#%%

d2 = np.array(list(rango))
values = np.array(values)
opt = values[:, 0]
g1 = values[:, 1]
g2 = values[:, 2]
l1 = values[:, 3]
l2 = values[:, 4]
l3 = values[:, 5]
nu = values[:, 6]
mu1 = values[:, 7]
mu2 = values[:, 8]
mu3 = values[:, 9]
mu4 = values[:, 10]
mu5 = values[:, 11]
mu6 = values[:, 12]
p1 = values[:, 13]
p2 = values[:, 14]
p3 = values[:, 15]
t = values[:, 16]

#%%
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4.25
plt.rcParams['font.size'] = 28
def plot(y, label, save=True):
    plt.figure(figsize=(20,12))
    plt.plot(d2, y)
    plt.xlabel(r'$d_2$')
    plt.ylabel(label)
    plt.grid(True)
    plt.title(f'Grafica de {label}$(d_2)$')
    if save:
        plt.savefig('plots/'+label[1:-1]+'.png')
plot(opt, label=r'$optimo$')
plot(g1, label=r'$g_1$')
plot(g2, label=r'$g_2$')
plot(p1, label=r'$p_1$')
plot(p2, label=r'$p_2$')
plot(p3, label=r'$p_3$')
plot(l3, label=r'$\lambda_1$')
plot(l2, label=r'$\lambda_2$')
plot(l3, label=r'$\lambda_3$')
plot(nu, label=r'$\nu$')
plot(mu1, label=r'$\mu_1$')
plot(mu2, label=r'$\mu_2$')
plot(mu3, label=r'$\mu_3$')
plot(mu4, label=r'$\mu_4$')
plot(mu5, label=r'$\mu_5$')
plot(mu6, label=r'$\mu_6$')
plot(t, label=r'$t$')