#%%
import numpy as np

def kroncker_product(A, B):
    m, n = A.shape
    p, q = B.shape
    AB = np.zeros([p*m, q*n])
    for i, element in enumerate(A.flatten()):
        mi = i % m
        ni = i // m
        AB[(mi * p):((mi + 1) * p), (ni * q):((ni + 1) * q)] = element * B

    return(AB)

def kroncker_sum(A, B):
    m, n = A.shape
    AB = kroncker_product(np.eye(n), A) + kroncker_product(B, np.eye(m))
    return(AB)

# %%
dt = 0.1
dx = 1
D1 = 1; D2 = 20;

# linear turning functions
def f(x):
    a = 0.05 ; b = 0.08
    return(a*x[:, :, 0] + b*x[:, :, 1])

def g(x):
    c = 0.1 ; d = -0.06
    return(c*x[:, :, 0] + d*x[:, :, 1])


def x_update(xt):
    D = np.array([[D1, 0], [0, D2]])
    xu = xt + (dt/dx**2)*(kroncker_product(D, L*xt[:, :, 1]) + f(xt[:, :, 1]))
    xv = xt + (dt/dx**2)*(kroncker_product(D, L*xt[:, :, 2]) + g(xt[:, :, 1]))
    xt_update = np.stack(xu, xv)
    return(xt_update)

#%%

steps = 1000

l = 5 # grid size n x n

L0 = np.diag(-2 * np.ones(l)) + np.diag(np.ones(l-1), 1) + np.diag(np.ones(l-1), -1)
L = kroncker_sum(L0, L0)

n = l**2
x = np.random.rand(n, n*2)

for step in range(steps):
    x = x_update(x)
# %%
