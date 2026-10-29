#%%
import sympy as sp
import numpy as np
from scipy.linalg import eigvals
import cmath

def linear(a, b, c, d, e, w):
    def f(u, v):
        return(a*u + b*v + c)

    def g(u, v):
        return(d*u + e*v + w)

    return f, g

def gray_scott(a, b):
    # u = autocatalyst, v = substrate, a = feed rate F, b = kill rate k (Pearson 1993)
    def f(u, v):
        return(u**2 * v - (a+b)*u)

    def g(u, v):
        return(-u**2*v+a*(1-v))
    
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

def bvam(eta, a, b, h, C):
    # Barrio-Varea-Aragon-Maini (1999): C sets the quadratic term, C = 0 -> stripes, larger C -> spots
    def f(u, v):
        return(eta*(u + a*v - C*u*v - u*v**2))

    def g(u, v):
        return(eta*(b*v + h*u + C*u*v + u*v**2))

    return f, g

def fitzhugh_nagumo(eps, a1, a0):
    # bistable when a1 > 1: stable states u ≈ ±1/√2 either side of a saddle at u ≈ 0
    # a0 tilts the balance between them: a0 < 0 favours low u (spots of high u), a0 > 0 the reverse
    def f(u, v):
        return(u - u**3 - v)

    def g(u, v):
        return(eps*(u - a1*v - a0))

    return f, g

def initial_fill(grid_size, fill=None, pert=False):
    initial = np.zeros((grid_size**2, 2))
    if fill != None:
        initial.fill(fill)

    if pert == True:
        initial += np.random.normal(loc=0.0, scale=1, size=initial.shape)
    return initial
# %%

def nonspacial_eigenvalues(f, g):
    # for no diffusion - 
    u = sp.symbols('u')
    v = sp.symbols('v')

    F = f(u, v)
    G = g(u, v)

    J = sp.Matrix([
        [sp.diff(F, u), sp.diff(F, v)],
        [sp.diff(G, u), sp.diff(G, v)]
    ])

    #jacobian_numeric = np.array(jacobian.subs({u: u0, v: v0})).astype(np.float64)
    eigenvalues = J.eigenvals()
    eigenvalues_array = np.array([float(ev.evalf().as_real_imag()[0]) for ev in eigenvalues])
    return eigenvalues_array


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
