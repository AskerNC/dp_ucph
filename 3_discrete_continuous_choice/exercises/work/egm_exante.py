# Import package and module
import numpy as np
import utility as util
import tools
from scipy import interpolate

def EGM(sol,t,par):
    #sol = EGM_loop(sol,t,par) 
    sol = EGM_vec(sol,t,par) 
    return sol

def EGM_loop (sol,t,par):

    c_plus_interp = interpolate.interp1d(sol.m[t+1,:],sol.c[t+1,:], bounds_error=False, fill_value = "extrapolate")
    for i_a,a in enumerate(par.grid_a[t,:]):
        

        if t+1<=par.Tr: # no pension in next period
            fac = par.G*par.L[t]*par.psi_vec
            inv_fac =1 /fac
            w = par.w

            
            # Futute m and c
            m_plus = inv_fac* par.R *a +par.xi_vec
            c_plus = c_plus_interp(m_plus)
        else:
            fac = par.G*par.L[t]
            inv_fac =1 /fac
            w=1

            
            # Futute m and c
            m_plus = inv_fac* par.R *a +1
            c_plus = c_plus_interp(m_plus)
        

        # Future marginal utility
        marg_u_plus = util.marg_util(fac*c_plus,par)
        avg_marg_u_plus = np.sum(w*marg_u_plus)

        # Currect C and m
        sol.c[t,i_a+1]=util.inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
        sol.m[t,i_a+1]=a+sol.c[t,i_a+1]
    
    return sol

def EGM_vec (sol,t,par):

    c_plus_interp = interpolate.interp1d(sol.m[t+1,:],sol.c[t+1,:], bounds_error=False, fill_value = "extrapolate")

    if t+1<=par.Tr: # no pension in next period
        fac = np.tile(par.G*par.L[t]*par.psi_vec,par.Na)
        xi = np.tile(par.xi_vec,par.Na)
        a = np.repeat(par.grid_a[t],par.Nshocks)
        
        w = np.tile(par.w,(par.Na,1))
        dim = par.Nshocks

        
    else:
        fac = par.G*par.L[t] *np.ones((par.Na))
        xi = np.ones((par.Na))
        a = par.grid_a[t,:]
        w=np.ones((par.Na,1))
        dim = 1

    inv_fac =1/fac

    # Futute m and c
    m_plus = inv_fac* par.R *a +xi
    c_plus = c_plus_interp(m_plus)
    #c_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c[t+1,:], m_plus)

    # Future marginal utility
    marg_u_plus = util.marg_util(fac*c_plus,par)
    marg_u_plus = np.reshape(marg_u_plus,(par.Na,dim))

    avg_marg_u_plus = np.sum(w*marg_u_plus,1)

    # Currect C and m
    sol.c[t,1:]=util.inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    sol.m[t,1:]= par.grid_a[t,:] + sol.c[t,1:]


    return sol
