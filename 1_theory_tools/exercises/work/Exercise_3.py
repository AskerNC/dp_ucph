# import packages used
import numpy as np
import scipy.optimize as optimize
from scipy import interpolate # Interpolation routines

def solve_consumption_grid_search(par):
     # initialize solution class
    class sol: pass
    sol.C = np.zeros(par.num_W)
    sol.V = np.zeros(par.num_W)
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C) 
    
    # Resource grid
    grid_W = par.grid_W

    # Init for VFI
    sol.delta = 1000 #difference between V_next and V_now
    sol.it = 0  #iteration counter 
    
    while (par.max_iter>= sol.it and par.tol<sol.delta):
        sol.it +=1
        V_next = sol.V.copy()
        interp = interpolate.interp1d(par.grid_W,V_next,bounds_error=False,fill_value='extrapolate')
        for iw,w in enumerate(grid_W):  # enumerate automaticcaly unpack w
            fun = lambda x: -V(x,w,interp,par)       
            
            res = optimize.minimize(fun,0.5,bounds =((1e-8,1-1e-8),))
            
            sol.V[iw] = -res.fun
            sol.C[iw] = res.x*w
            
            
        sol.delta = np.amax(np.abs(sol.V - V_next))
    
    return sol


def solve_consumption_grid_search2(par):
    
     # initialize solution class
    class sol: pass
    sol.C = np.zeros(par.num_W)
    sol.V = np.zeros(par.num_W)
    
    # consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C) 
    
    # Resource grid
    grid_W = par.grid_W

    # Init for VFI
    sol.delta = 1000 #difference between V_next and V_now
    sol.it = 0  #iteration counter 
    
    while (par.max_iter>= sol.it and par.tol<sol.delta):
        sol.it +=1
        V_next = sol.V.copy()
        for iw,w in enumerate(grid_W):  # enumerate automaticcaly unpack w
            c = grid_C*w

            wt1 = w-c
            V_guess = np.sqrt(c)+par.beta * np.interp(wt1,grid_W,V_next)

            i = np.argmax(V_guess)
            

            sol.V[iw] = V_guess[i]
            sol.C[iw] = c[i]
            
            
        sol.delta = np.amax(np.abs(sol.V - V_next))
    
    return sol


def V(x,w,interp,par):
    #"unpack" c
    if type(x) == np.ndarray: # vector-type: depends on the type of solver used
        c = w*x[0] 
    else:
        c = w*x

    wt1 = w-c

    return np.sqrt(c)+par.beta*interp(wt1)