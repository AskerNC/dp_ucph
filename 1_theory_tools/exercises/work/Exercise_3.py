# import packages used
import numpy as np
import scipy.optimize as optimize

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
        for iw,w in enumerate(grid_W):  # enumerate automaticcaly unpack w
            fun = lambda x: -V(x,w,V_next,grid_W,par)       
            
            res = optimize.minimize(fun,sol.C[iw],bounds =((1e-4,1-1e-4),))
            
            sol.V[iw] = -res.fun
            sol.C[iw] = res.x
            
            
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


def V(x,w,V_vec,grid_W,par):
    #"unpack" c
    if type(x) == np.ndarray: # vector-type: depends on the type of solver used
        c = w*x[0] 
    else:
        c = w*x

    wt1 = w-c

    Vt1 = np.interp(wt1,grid_W,V_vec)

    return np.sqrt(c)+par.beta*Vt1 