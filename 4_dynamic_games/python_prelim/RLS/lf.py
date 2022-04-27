import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import copy
from types import SimpleNamespace


class Lf_model():

    def __init__(self,dict_={}) :
            
        self.maxc= 5
        self.minc =0
        self.nC = 4 # number of states in tech process
        
        
        self.k0  = 0
        self.k1  = 8.3
        self.k2  = 1
        self.R   = 0.05
        self.Dt  = 1
        self.pf = 1
        
        self.Kform = 'smooth'


        for key,value in dict_.items():
            setattr(self,key,value)

        # update dependent parameters 
        self.C = np.linspace(self.maxc,self.minc,self.nC)

        self.p = np.ones((self.nC,1)) * self.pf 
        self.p[-1] = 0 

        self.beta = np.exp(-self.R*self.Dt)
        self.T = (self.nC-1)*3+1

        self.nESS = self.nC * (self.nC+1) * (self.nC+1) /6



        # It should be notes that python arrays are structed as: layers, rows, columns
        # while matlab is structures as rows, columns, layers
        self.firm1 = np.empty((self.nC,self.nC,self.nC))
        self.firm1[:,:,:] =np.nan

        self.firm2 =copy.copy(self.firm1)

        # fill out firms costs in different layers
        for i in range(self.nC):
            x = np.tile(self.C[0:i+1], (i+1, 1))
            self.firm1[i,0:i+1,0:i+1]= x.T

            self.firm2[i,0:i+1,0:i+1]=x
       
        # create stage_index
        self.rlsIndex()



    def rlsIndex(self):
        Ns =   np.sum( (np.array(range(self.nC)) +1)**2)
        T = 1 + 3 * (self.nC-1)

        Ns_in_stages = np.ones((1,T))

        j = 0
        l = self.nC
        while l>1:
            Ns_in_stages[0,j] = 1
            j += 1
            Ns_in_stages[0,j]  = 2* (l-1)
            j += 1 
            Ns_in_stages[0,j] = (l-1)**2
            j +=1 
            l -= 1
        
        self.stage_index = np.cumsum(Ns_in_stages)


    def cSS(self):
        # OUTPUT: 1 x N struct tau representing stages of state space
		# PURPOSE: Create state space structure to hold info on identified
		# equilibria


		# P1 player 1 probability invest
		# vN1 value of not investing for player 1
		# vI1 value of investing for player 1
		# P2 player 2 probability of invest
		# vN2 value of not investing for player 2
		# vI2 value of investing for player 2 	    
		# Initialize datastructure for the state space
        eq= SimpleNamespace(P1=[],vN1=[],vI1=[],P2=[],vN2=[],vI2=[])


        self.tau =[]
        for i in range(self.nC):
            pass




    def K(self,c):
        if self.Kform=='smooth':
            return self.k0+self.k1/(1+self.k2*c)
        elif self.Kform=='constant':
            return self.k1

    def payoff(self,x):
        return (self.p-x) * self.Dt

    def r1(self,x1,x2):
        return np.max((x2-x1,0))
    
    def r2(self,x1,x2):
        return np.max((x1-x2,0))

    def Phi(self,vN,vI):
        return np.fmax(vN,vI) 



    def cESS(self):
        '''
        Create N x N x N array ess.index
		 PURPOSE:
		 ess.index(m,n,h) --> j
		 where j is the index for ess.esr such that
		 ess.esr(j)+1 is the equilibrium number played in state space
		 point (m,n,h) this equilibrium is stored in the ss-object as
		 ss(h).(m,n,j).eq = ss(h).(m,n,ess.esr(ess.index(m,n,h))+1)
        '''
        n = self.nC

        ess = SimpleNamespace()

        ess.index = np.empty((n,n,n))
        ess.index[:,:,:] =np.nan

        for ic in range(n):
            for ic1 in range(ic+1):
                for ic2 in range(ic+1):
                    #print(ic,ic1,ic2)
                    ess.index[ic,ic1,ic2] = self.essindex(n,ic1+1,ic2+1,ic+1)

        #  N*(N+1)*(2*N+1)/6 = sum(1^2 + 2^2 + 3^2 + ... + N^2)
        shape =(1,n*(n+1)*(2*n+1)//6 )
        ess.esr = np.zeros(shape)
        ess.bases = np.ones(shape)

        self.ess = ess 
        


    def essindex(self,x,ic1,ic2,ic):
        # INPUT: x is count of technological levels
	    # OUTPUT: ess index number for point (m,n,h) i state space

        if np.all([ic1,ic2]==[ic,ic]):
            index = 1+ x*(x+1)*(2*x+1) //6 - ic*(ic+1)*(2*ic+1)//6
        elif ic2==ic:
            index = 1+ x*(x+1)*(2*x+1) //6 - ic*(ic+1)*(2*ic+1)//6 +ic1
        elif ic1==ic:
            index = 1+ x*(x+1)*(2*x+1) //6 - ic*(ic+1)*(2*ic+1)//6 + ic -1 +ic2
        else:
            #print(ic,ic1,ic2)
            index = 1+ x*(x+1)*(2*x+1) //6 - ic*(ic+1)*(2*ic+1)//6 + 2 *(ic-1) + np.ravel_multi_index((ic2-1,ic1-1),(ic,ic)) + 1 - 1* (ic2-1) 
        
        return index 