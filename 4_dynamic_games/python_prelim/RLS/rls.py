from random import triangular
from types import SimpleNamespace
import numpy as np
import copy


def solve(G,lf,ESS0,stage_index ,maxEQ=50000, printev=500):
    '''
        maxEQ  =  50000 maximum number of iterations
		printev  = 500;  print every rlsp.print equilibria (0: no print, 1: print every, 2: print every second)
    '''
    rlsp = SimpleNamespace(maxEQ=maxEQ, printev=printev )

    TAU = np.empty((rlsp.maxEQ,))
    TAU[:]= np.nan
    out = np.empty(((rlsp.maxEQ,)),dtype=object)
    tau = np.size(stage_index)
    ESS = np.empty((rlsp.maxEQ+1,), dtype=object)
    iEQ= 1

    ESS[0] = ESS0
    

    while iEQ <= rlsp.maxEQ:
        TAU[iEQ-1]= tau
        #print(iEQ)
        #print(tau)
        #print(ESS[iEQ-1])

        ESS[iEQ-1] = G(lf,ESS[iEQ-1],tau) 

        if iEQ%rlsp.printev==0:
            print(f'ESR({iEQ}).esr      : [')
            print(ESS[iEQ-1].esr)
            print(']')
            print(f'ESR({iEQ}).bases    : [')
            print(ESS[iEQ-1].bases)
            print(']')
        

        out[iEQ-1]= output(lf)

        ESS[iEQ-1+1 ] = addOne(ESS[iEQ-1])
        
        #change_index = np.min( np.nonzero(ESS[iEQ-1+1].esr - ESS[iEQ-1].esr != 0 )) 
        change_index = np.min( np.nonzero( (ESS[iEQ-1+1].esr - ESS[iEQ-1].esr != 0 ) .flatten() )[0] )
        #change_index = np.min( np.nonzero( (ESS[1-1+1].esr - ESS[1-1].esr != 0 ) .flatten() )[0] )
        tau = np.sum( change_index+1 <= stage_index)-1
        #print(tau)


        if np.all(ESS[iEQ-1+1].esr == -1):
            break
        
        iEQ += 1 

    TAU = TAU[0:iEQ]
    ESS = ESS[0:iEQ+1]
    out = out[0:iEQ]


    return ESS, TAU, out


def output(lf):
    out = SimpleNamespace()
    out.MPEesr = lf.ess.esr
    eq  = lf.ss[0].EQs[0,0,0]
    out.V1 = np.fmax(eq.vN1, eq.vI1)
    out.V2 = np.fmax(eq.vN2, eq.vI2)
    return out


def addOne(ess_):
    ess = copy.copy(ess_)
    n = len(ess.esr)
    X = np.zeros((n,), dtype=int)
    R =1 

    for i in range(n-1,-1,-1):
        X[i] = (ess.esr[i] + R) % ess.bases[i]

        R = (ess.esr[i]+R )// ess.bases[i]

    if R>0:
        #When exiting the loop R > 0 occurs when all ESS.number is max allowed
		# which is 1 below the base.
		# println("No more equilibria to check.")
        ess.esr = -1*np.ones((n,1), dtype=int)
    
    else:
        ess.esr = X

    return ess




