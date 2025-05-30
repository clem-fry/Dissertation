#%%
import sympy as sp
import numpy as np
from scipy.linalg import eigvals
import cmath

def linear(a, b, e, d, c=0, w=0):
    def f(u, v):
        return(a*u + b*v + c)

    def g(u, v):
        return(d*u + e*v + w)

    return f, g

def gray_scott(a, b):
    def f(u, v):
        return(u**2 * v - (a+b)*u)

    def g(u, v):
        return(-u**2*v+a*(1+v))
    
    return f, g

def geirer_meinhardt(a, b, c):
    def f(u, v):
        return (a - b*u + u**2/v)
    
    def g(u, v):
        return (u**2 - c*v)
    
    return f, g

def schnakenberg(a, b):
    def f(u, v):
        return(a-u+u**2*v)
    
    def g(u, v):
        return(b-u**2*v)
    
    return f, g
# %%

def dispersion(f, g, u0, v0, Du, Dv, k):
    # for no diffusion - 
    u = sp.symbols('u')
    v = sp.symbols('v')

    F = f(u, v)
    G = g(u, v)

    J = sp.Matrix([
        [sp.diff(F, u), sp.diff(F, v)],
        [sp.diff(G, u), sp.diff(G, v)]
    ])

    J_sub = J.subs({u: u0, v: v0})

    J_dispersion = sp.Matrix([
        [J_sub[0, 0] - Du * k**2, J_sub[0, 1]],
        [J_sub[1, 0], J_sub[1, 1] - Dv * k**2]
    ])

    #jacobian_numeric = np.array(jacobian.subs({u: u0, v: v0})).astype(np.float64)
    eigenvalues = J_dispersion.eigenvals()
    eigenvalues_array = np.array([float(ev.evalf().as_real_imag()[0]) for ev in eigenvalues])
    J_numeric = np.array(J_dispersion).astype(np.float64)
    return eigenvalues_array, J_numeric
# %%
