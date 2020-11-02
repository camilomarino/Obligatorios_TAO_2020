import cvxpy as cp
import numpy as np
from utils import get_Pi, get_qi, get_ri, get_constrain_i, get_constrain_i_value
from utils import P0, q0, r0, N, f

#%%

# PC

x = cp.Variable(N)

objective = cp.Minimize(f(x))


constraints = [get_constrain_i(x, i, N) for i in range(N)]
prob = cp.Problem(objective, constraints)

result = prob.solve()

xpc = x.value
print('xpc* = ', xpc)
print('fpc* = ', result)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print('mu_pc* = ', [float(constraints[i].dual_value) for i in range(N)])
print()

#%%
# PA
xpa = np.ones_like(xpc)
xpa[xpc<0] = -1
print('xpa* = ', xpa)
print('fpa* = ', f(xpa).value)


#%%

