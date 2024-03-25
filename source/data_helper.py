import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import pickle

# Method of manufactured soltuions (MMS)
class pde_exact():
    def __init__(self):
        ### Symbolic form for MMS
        # 1D Poisson
        x, c1, c2, c3, c4, c5 = sy.symbols('x, c1, c2, c3, c4, c5')
        u = c1*sy.sin(1*x*sy.pi) + c2*sy.sin(2*x*sy.pi) + c3*sy.sin(3*x*sy.pi) + c4*sy.sin(4*x*sy.pi) + c5*sy.sin(5*x*sy.pi)
        
        eq = u.diff(x).diff(x)
        self.poisson_f_func = sy.lambdify([x, c1, c2, c3, c4, c5],eq,'numpy')
        self.poisson_u_func = sy.lambdify([x, c1, c2, c3, c4, c5],u,'numpy')
        
        # 2D Allen-Cahn
        lam = 0.1
        x, y, c1, c2, c3, c4, c5 = sy.symbols('x, y, c1, c2, c3, c4, c5')
        u = c1*sy.sin(1*x*sy.pi)*sy.sin(1*y*sy.pi) + c2*sy.sin(2*x*sy.pi)*sy.sin(2*y*sy.pi) + c3*sy.sin(3*x*sy.pi)*sy.sin(3*y*sy.pi) + c4*sy.sin(4*x*sy.pi)*sy.sin(4*y*sy.pi) + c5*sy.sin(5*x*sy.pi)*sy.sin(5*y*sy.pi)
        eq = lam*(u.diff(x).diff(x) + u.diff(y).diff(y)) + u*(u**2 - 1)
        self.allencahn_f_func = sy.lambdify([x, y, c1, c2, c3, c4, c5],eq,'numpy')
        self.allencahn_u_func = sy.lambdify([x, y, c1, c2, c3, c4, c5],u,'numpy')
        
    def poisson_1d(self, c1, c2, c3, c4, c5, N):
        x_val = np.linspace(-1,1,N)
        u_vals = self.poisson_u_func(x_val, c1, c2, c3, c4, c5)
        f_vals = self.poisson_f_func(x_val, c1, c2, c3, c4, c5)
        return u_vals, f_vals
    
    def poisson_1d_Fdraw(self, c1, c2, c3, c4, c5, N):
        x_val = np.linspace(-1,1,N)
        u_vals = self.poisson_u_func(x_val, c1, c2, c3, c4, c5)
        f_vals = self.poisson_f_func(x_val, c1, c2, c3, c4, c5)
        return u_vals, f_vals
    
    def diffreac_1d_IC(self, c1, c2, c3, c4, c5, N):
        x = np.linspace(-1,1,N)
        u0 = (x**2-1)/5 * (c1*(np.cos(1*x)**2 - 1) + c2*(np.cos(2*x)**2 - 1) + c3*(np.cos(3*x)**2 - 1) + c4*(np.cos(4*x)**2 - 1) + c5*(np.cos(5*x)**2 - 1))
        return u0

    def allen_cahn_2d(self, c1, c2, c3, c4, c5, N, M):
        x_val = np.linspace(0,1,N); y_val = np.linspace(0,1,M)
        f_vals = np.zeros((x_val.size,y_val.size))
        u_vals = np.zeros((x_val.size,y_val.size))

        for i in range(x_val.size):
            u_vals[i,:] = self.allencahn_u_func(x_val[i],y_val,c1, c2, c3, c4, c5)
            f_vals[i,:] = self.allencahn_f_func(x_val[i],y_val,c1, c2, c3, c4, c5)
        return u_vals, f_vals
    
def fisher_solver(u0, fidelity_x,fidelity_t):
    D = 0.1; k = 0.1;
    x = np.linspace(-1,1,fidelity_x)
    t = np.linspace(0,1,fidelity_t)
    u = np.zeros((fidelity_t+1,fidelity_x))
    u[0,:] = u0
    u[1,:] = u0
    dx = x[1]-x[0]
    dt = t[1]-t[0] 
    for n in range(1,fidelity_t): # temporal loop
        a = np.zeros(fidelity_x); b = np.zeros(fidelity_x); c = np.zeros(fidelity_x); d = np.zeros(fidelity_x)
        for i in range(0,fidelity_x): # spatial loop
            # Create vectors for a, b, c, d
            a[i] = -D*dt
            b[i] = dx**2 - dx**2 * dt * k * (1-(2*u[n,i]-u[n-1,i])) + 2 * dt * D
            c[i] = -D*dt
            d[i] = (dx**2)*u[n,i]
        # BC's
        a[0] = 0; b[0] = 1; c[0] = 0; d[0] = 0 # Dirichlet Condition
        a[-1] = 0; b[-1] = 1; c[-1] = 0; d[-1] = 0
        # Solve
        u[n+1,:] = thomas_alg(a,b,c,d)
    v = u[1:,:]
    return v, x, t

def thomas_alg(a, b, c, d):
    n = len(b)
    x = np.zeros(n)
    for k in range(1,n):
        q = a[k]/b[k-1]
        b[k] = b[k] - c[k-1]*q
        d[k] = d[k] - d[k-1]*q
    q = d[n-1]/b[n-1]
    x[n-1] = q
    for k in range(n-2,-1,-1):
        q = (d[k]-c[k]*q)/b[k]
        x[k] = q
    return x
