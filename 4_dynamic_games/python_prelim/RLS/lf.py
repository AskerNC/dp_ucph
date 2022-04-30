from msilib.schema import Error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
from scipy import optimize
import copy, math
from types import SimpleNamespace
import pandas as pd

#import pandas as pd

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

        # create empty containers for eq
        self.cSS()

        # create linear index 
        self.cESS()

        # ESS, created by cESS, contains equilbriums for one world, we need a list of possible equilibria that we can then loop through
        



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

        

        self.ss =[SimpleNamespace() for i in range(self.nC)]
        for i in range(self.nC):

            EQs = np.empty((5,i+1,i+1),dtype=object)

            ## allow for 5 equilibria, but there need to be at least one:
            EQs[0,:,:]= SimpleNamespace(P1=[],vN1=[],vI1=[],P2=[],vN2=[],vI2=[])
            
            self.ss[i].EQs = EQs # container for identified equilibriums
            self.ss[i].nEQ = np.zeros((i+1,i+1), dtype=int) #container for number of eqs in (x1,x2,c) point


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

        ess.index = np.empty((n,n,n),dtype=object)
        ess.index[:,:,:] =np.nan

        for ic in range(n):
            for ic1 in range(ic+1):
                for ic2 in range(ic+1):
                    #print(ic,ic1,ic2)
                    ess.index[ic,ic1,ic2] = self.essindex(n,ic1+1,ic2+1,ic+1)

        #  N*(N+1)*(2*N+1)/6 = sum(1^2 + 2^2 + 3^2 + ... + N^2)
        shape =(n*(n+1)*(2*n+1)//6 ,)
        ess.esr = np.zeros(shape,dtype=int)
        ess.bases = np.ones(shape, dtype=int)

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



### useful functions

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


    def quad(self,a,b,c):
        #Solves:  ax^2  + bx + c = 0
		# but also always return 0 and 1 as candidates for probability of
		# investment

        d = b**2 - 4*a*c


        if abs(a) < 1e-8:
            pstar =  [0. , 1., -c/b]
        else:
            if d < 0:
                pstar = [0. ,1.]
            elif d == 0:
                pstar = [0. ,1. , -b/(2*a)];
            else:
                pstar = [0. ,1. , (-b - np.sqrt(d))/(2*a),  (-b + np.sqrt(d))/(2*a)]

        return np.array(pstar, dtype=object)






    ##### solving the model 


    def state_recursion(self,tau):

        if tau == self.T:
            self.solve_last_corner()
            tau = tau-1
        
        if tau == self.T-1:
            self.solve_last_edge()
            tau = tau-1 
        
        if tau == self.T-2:
            self.solve_last_interior()
            tau = tau-1
        
        
        while tau!=0: # infinite loop that breaks on tau=0
            if tau%3==1:
                ic = math.ceil((tau+2)/3)-1
                self.solve_corner(ic)
                tau = tau-1
                if tau ==0: 
                    break
            
            if tau%3==0:
                ic = math.ceil((tau+2)/3)-1
                self.solve_edge(ic)
                tau = tau-1
            
            if tau%3==2:
                ic = math.ceil((tau+2)/3)-1
                self.solve_interior(ic)
                tau = tau-1





    ## last 
    def solve_last_corner(self):
        #  output: Equilibrium of state space point (h,h,h) with h = mp.nC

        h = self.nC -1 # number of tech levels (minus 1 for index )
        c = self.minc # State of the art marginal cost for last tech. level


        # Both players have state of the art technology implemented ic1=ic2=c
        # If K>0 the vN1 = r1/(1-beta) .... geometric sum
        vN1 = (self.r1(c,c)+self.beta * max(0,-self.K(c)))  /  (1-self.beta)

        vI1 = vN1 - self.K(c)
        P1 = vI1 > vN1;  #  Equivalent to 0>mp.K(c)
    
        vN2 = (self.r2(c,c)+self.beta * max(0 , -self.K(c)))  /  (1-self.beta)
        vI2 = vN2 - self.K(c)
        P2 = vI2 > vN2  # Equivalent to 0>mp.K(c) and hence equal to P1;

        self.ss[h].EQs[0,h,h] = SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2);
        # Only one equilibrium is possible
        self.ss[h].nEQ[h,h]=1
        self.ess.bases[self.ess.index[h,h,h] -1 ]= 1 


    def solve_last_edge(self):
        '''
        OUTPUT:
			% Equilibria lf.EQ(P1, vN1, vI1, P2, vN2, vI2) for edge state space points
			% of the final layer:
			% Final layer <=> s=(x1,x2,c) with c = min(mp.C) 
			% Edge <=> s=(x1,x2,c) with x2 = c = min(mp.C) and x1 > c or
			% s=(x1,x2,c) with x1 = c = min(mp.C) and x2 > c    

        '''
        ic = self.nC -1# Get the level of technology final layer
        c = self.minc # Get state of the art marginal cost for tech. of final layer

        h = 0 
        # h is used to select equilibria in the corner of the final layer but there
        # is only ever 1 equilibria in the corner
        # If we did not apply this apriori knowledge we would have to use ESS
        

        # Get the value of max choice in the corner of final layer s = (c,c,c)
        g1_ccc = max(self.ss[ic].EQs[h,ic,ic].vN1,self.ss[ic].EQs[h,ic,ic].vI1);
        g2_ccc = max(self.ss[ic].EQs[h,ic,ic].vN2,self.ss[ic].EQs[h,ic,ic].vI2);

        #  Player 2 is at the edge s=(x1,x2,c) with x2=c=min(mp.C) and x1>c
        for ic1 in range(ic):
            
            x1 = self.C[ic1]
            vI1 = self.r1(x1,c) - self.K(c) + self.beta* g1_ccc


            vN1search = lambda z : self.r1(x1,c) + self.beta * self.Phi(z,vI1) - z

            res = optimize.root(vN1search,x0 = vI1)
            vN1 = res.x

            P1 = vI1 > vN1


            vN2 = ( self.r2(x1,c) + self.beta * (P1*g2_ccc+(1-P1)*self.Phi(0,-self.K(c))) )  /  ( 1-self.beta*(1-P1) )
            vI2 = vN2 - self.K(c)
            P2 = vI2 > vN2;

            self.ss[ic].EQs[h,ic1,ic] =  SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2)
            
            # Only one equilibrium is possible
            self.ss[ic].nEQ[ic1,ic]=1
            self.ess.bases[self.ess.index[ic,ic1,ic1] -1]= 1 

        # Player 1 is at the edge s=(x1,x2,c) with x1=c=min(mp.C) and x2>c
        for ic2 in range(ic):
            
            x2 = self.C[ic2]
            vI2 = self.r2(c,x2) - self.K(c) + self.beta* g2_ccc

            vN2search = lambda x : self.r2(c,x2) + self.beta * self.Phi(x,vI2) - x

            res = optimize.root(vN2search,x0 = vI2)
            vN2 = res.x

            P2 = vI2 > vN2


            vN1 = ( self.r1(c,x2) + self.beta * (P2*g1_ccc+(1-P2)*self.Phi(0,-self.K(c))) )  /  ( 1-self.beta*(1-P2) )
            vI1 = vN1 - self.K(c)
            P1 = vI1 > vN1;


            self.ss[ic].EQs[h,ic,ic2] =  SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2)
            
            # Only one equilibrium is possible
            self.ss[ic].nEQ[ic,ic2]=1
            self.ess.bases[self.ess.index[ic,ic,ic2]-1]= 1 


    def solve_last_interior(self):

        # outside loop
        ic =self.nC-1
        c= self.C[ic]

        def g1( iC1 , iC2, iC):
            eq1 =  self.ss[iC].EQs[self.ess.esr[self.ess.index[iC,iC1,iC2] -1] , iC1, iC2] 
            
            return np.maximum( eq1.vN1, eq1.vI1)
        def g2( iC1 , iC2, iC):
            eq1 =  self.ss[iC].EQs[self.ess.esr[self.ess.index[iC,iC1,iC2] -1], iC1, iC2] 
            return np.maximum( eq1.vN2, eq1.vI2)
        

        for ic1 in range(ic):
            for ic2 in range(ic):
                # Player 1 -> leads to P2 candidates
                a = self.r1(self.C[ic1], self.C[ic2]) - self.K(c) + self.beta*g1(ic, ic2, ic) 
                b = self.beta*(g1(ic, ic, ic)-g1(ic, ic2, ic))
                d = self.r1(self.C[ic1],self.C[ic2]);
                e = self.beta*g1(ic1, ic, ic);

                    
                b_0 = - self.beta * b 
                b_1 = self.beta * g1(ic1, ic, ic) + (self.beta-1)*b - self.beta*a
                b_2 = self.r1(self.C[ic1],self.C[ic2] ) + (self.beta-1) * a


                pstar2 = self.quad(b_0, b_1, b_2)

                # always return 1 and 0 for the pure strategies


                # Player 2 -> leads to P1 candidates
                A = self.r2(self.C[ic1], self.C[ic2]) - self.K(c) + self.beta*g2(ic1, ic, ic); 
                B = self.beta*(g2(ic, ic, ic)-g2(ic1, ic, ic));
                D = self.r2(self.C[ic1],self.C[ic2]);
                E = self.beta*g2(ic, ic2, ic);

                d_0 = - self.beta * B;
                d_1 = self.beta*g2(ic, ic2, ic) + (self.beta-1) * B - self.beta*A;
                d_2 = self.r2(self.C[ic1],self.C[ic2]) + (self.beta-1) * A;

                pstar1 = self.quad(d_0, d_1, d_2);
                    
                # Find equilibria based on candidates
                # Number of equilibria found are 0 to begin with

                count = 0
                for i in range(len(pstar1)):
                    for j  in range(len(pstar2)):
                        if np.all( [k in [0,1] for k in [i,j] ] ): # these are pure strategies
                            # If the polynomial is negative vI > vN
                            # hence player invests set exPj=1 else 0
                            # exP1 is best response to pstar2(j)
                            exP1 = b_2 + b_1 * pstar2[j] + b_0 * pstar2[j]**2 < 0 
                            exP2 = d_2 + d_1 * pstar1[i] + d_0 * pstar1[i]**2 < 0 

                            # check if both are playing best response
                            # in pure strategies. Players best response
                            # should be equal to the candidate to which
                            # the other player is best responding.
                            if np.abs(exP1 - pstar1[i]) < 1e-8 and np.abs(exP2-pstar2[j]) < 1e-8:
                                # if exP1=0 and pstar_i=0 true
                                # if exP1=1 and pstar_i=1 true
                                # Testing whether best response exP1 is
                                # equal to pstar1(i) to which Player 2
                                # is best responding ...
                                count = count + 1;
                                vI1 = a + b*pstar2[j]; 
                                vN1 = (d + e*pstar2[j] + self.beta*(1-pstar2[j])*(a+b*pstar2[j]))*pstar1[i]     +     (1-pstar1[i])*(d+e*pstar2[j])/(1-self.beta*(1-pstar2[j]))
                                vI2 = A + B*pstar1[i]; 
                                vN2 = (D + E*pstar1[j] + self.beta*(1-pstar1[i])*(A+B*pstar1[i]))*pstar2[j]     +     (1-pstar2[j])*(D+E*pstar1[i])/(1-self.beta*(1-pstar1[i]))
                                

                                self.ss[ic].EQs[count-1, ic1, ic2] = SimpleNamespace(P1=pstar1[i],vN1=vN1,vI1= vI1,P2=pstar2[j],vN2=vN2,vI2= vI2)

                                

                        elif i > 1 and j > 1 and pstar1[i] >= 0 and pstar2[j] >= 0 and pstar1[i] <= 1 and pstar2[j] <= 1:
                            count = count + 1
                            v1 = a + b * pstar2[j]
                            v2 = A + B * pstar1[i]
                            self.ss[ic].EQs[count-1, ic1, ic2] = SimpleNamespace(P1=pstar1[i],vN1=v1,vI1= v1,P2=pstar2[j],vN2=v2,vI2= v2)

                self.ss[ic].nEQ[ic1,ic2] = count
                self.ess.bases[self.ess.index[ic,ic1,ic2] -1]= count



    ## upper layers 
    def solve_corner(self,ic):
        # ic is the index of the layer 
        
        c = self.C[ic]

        # propability of techonological development
        p = self.p[ic]


        # find index for equilibrium selection h=1 for simple selection rule
        # Need ic+1 because ss(ic+1).EQs(ic,ic,h).eq is to be accessed
        h = self.ess.esr[self.ess.index[ic+1,ic,ic]-1]

        eq = self.ss[ic+1].EQs[h,ic,ic]
        '''
        try:
            vN1 = (self.r1(c,c)+self.beta * p* np.maximum(eq.vN1, eq.vI1) + self.beta * (1-p) * np.maximum(0,-self.K(c) ) )  /  (1- (1-p)*self.beta)
        except:
            print('Error!:')
            print('h','ic')
            print(h,ic)
            self.ess.esr
            self.ess.index
            sdfs
            
        '''
        vN1 = (self.r1(c,c)+self.beta * p* np.maximum(eq.vN1, eq.vI1) + self.beta * (1-p) * np.maximum(0,-self.K(c) ) )  /  (1- (1-p)*self.beta)

        vI1 = vN1 - self.K(c)

        vN2 = (self.r2(c,c)+self.beta * p* np.maximum(eq.vN2, eq.vI2) + self.beta * (1-p) * np.maximum(0,-self.K(c) ) )  /  (1- (1-p)*self.beta)
        vI2 = vN2 - self.K(c)

        P1 = vI1 > vN1;  #  Equivalent to 0>mp.K(c)
        P2 = vI2 > vN2  # Equivalent to 0>mp.K(c) and hence equal to P1;



        self.ss[ic].EQs[0,ic,ic] = SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2);
        
        # Only one equilibrium is possible
        # No update of ESS.bases is necessary in principle: "there can BE ONLY ONE
	    # equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs+

        self.ss[ic].nEQ[ic,ic]=1
        self.ess.bases[self.ess.index[ic,ic,ic] -1 ]= 1 




    def solve_edge(self,ic):
        
        c = self.C[ic]

        # propability of techonological development
        p = self.p[ic]


        def H1(iC1, iC2, iC):
            index_p  = self.ess.esr[self.ess.index[iC+1,iC1,iC2]-1]
            index  = self.ess.esr[self.ess.index[iC,iC1,iC2]-1]
            
            eq_p = self.ss[iC+1].EQs[index_p,iC1, iC2]
            eq   = self.ss[iC  ].EQs[index,iC1, iC2]
            
            
            return p* self.Phi(eq_p.vN1,eq_p.vI1) + (1-p) *self.Phi(eq.vN1,eq.vI1)
            
        def H2(iC1, iC2, iC):
            index_p  = self.ess.esr[self.ess.index[iC+1,iC1,iC2]-1]
            index    = self.ess.esr[self.ess.index[iC  ,iC1,iC2]-1]

            eq_p = self.ss[iC+1].EQs[index_p,iC1, iC2]
            eq   = self.ss[iC  ].EQs[index,iC1, iC2]
            return p* self.Phi(eq_p.vN2,eq_p.vI2) + (1-p) *self.Phi(eq.vN2,eq.vI2) 

        #Efficiency ... why evaluate the call for each run of following loop? i is
		# constant in domain outside loop!! What changes in the function is the ESS.
    
        #  Player 2 is at the edge 
        for ic1 in range(ic):
            
            c1 = self.C[ic1]
            
            

            vI1 = self.r1(c1,c) - self.K(c) + self.beta* H1(ic,ic,ic)

            index = self.ess.esr[self.ess.index[ic+1,ic1,ic]-1 ]
            eq_p = self.ss[ic+1].EQs[index,ic1,ic]
            
            def vN1search(z):
                return self.r1(c1,c) + self.beta * ( p * ( np.maximum( eq_p.vN1,eq_p.vI1 ) ) + (1-p)* np.maximum(z,vI1) ) - z 
                

            res = optimize.root(vN1search,x0 = vI1)
            vN1 = res.x

            P1 = vI1 > vN1

            
            vN2 = ( self.r2(c1,c) + self.beta * (P1*H2(ic,ic,ic)+(1-P1)*(p*self.Phi(eq_p.vN2,eq_p.vI2) + (1-p) * self.Phi(0,-self.K(c)) ) ) )   /  ( 1-self.beta*(1-P1)*(1-p) )
            vI2 = vN2 - self.K(c)
            P2 = vI2 > vN2;

            self.ss[ic].EQs[0,ic1,ic] =  SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2)
            
            # Only one equilibrium is possible
            self.ss[ic].nEQ[ic1,ic]=1
            self.ess.bases[self.ess.index[ic,ic1,ic] -1]= 1 


            
        # Player 1 is at the edge s=(x1,x2,c) with x1=c=min(mp.C) and x2>c
        for ic2 in range(ic):
            
            c2 = self.C[ic2]
            vI2 = self.r2(c,c2) - self.K(c) + self.beta* H2(ic,ic,ic)

            index = self.ess.esr[self.ess.index[ic+1,ic,ic2]-1 ]
            eq_p = self.ss[ic+1].EQs[index,ic,ic2]
            
            def vN2search(z):
                return self.r2(c,c2) + self.beta * ( p * ( np.maximum( eq_p.vN2,eq_p.vI2 ) ) + (1-p)* np.maximum(z,vI2) ) - z 
                

            res = optimize.root(vN2search,x0 = vI2)
            vN2 = res.x

            P2 = vI2 > vN2

            
            vN1 = ( self.r1(c,c2) + self.beta * (P2*H2(ic,ic,ic)+(1-P2)*(p*self.Phi(eq_p.vN1,eq_p.vI1) + (1-p) * self.Phi(0,-self.K(c)) ) ) )   /  ( 1-self.beta*(1-P2)*(1-p) )
            vI1 = vN1 - self.K(c)
            
            P1 = vI1 > vN1

            self.ss[ic].EQs[0,ic,ic2] =  SimpleNamespace(P1=P1,vN1=vN1,vI1= vI1,P2=P2,vN2=vN2,vI2= vI2)
            
            # Only one equilibrium is possible
            self.ss[ic].nEQ[ic,ic2]=1
            self.ess.bases[self.ess.index[ic,ic,ic2] -1]= 1  
            # No update of ESS.bases is necessary: "there can BE ONLY ONE
			# equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs



    def solve_interior(self,ic):
        #ss is state space structure with solutions for final layer edge and
        # corner
        # ic is the level of technology for which to solve
        # ESS is struc with information holding ESS.esr being equilibrium selection
        # rule and ESS.bases being the bases of the ESS.esr's
        c=self.C[ic]

        for ic1 in range(ic):
            for ic2 in range(ic):
                self.find_interior(ic1,ic2,ic,c)



    def find_interior(self,ic1,ic2,ic,c):
        p = self.p[ic]
        q=1-p

        # h is used for selected equilibrium in state realized when technology
        # develops hence ic+1 in ESS.index(ic1,ic2,ic+1)
        h = self.ess.esr[self.ess.index[ic+1,ic1,ic2]-1]
        
            
        def H1(iC1, iC2, iC):
            index_p  = self.ess.esr[self.ess.index[iC+1,iC1,iC2]-1]
            index    = self.ess.esr[self.ess.index[iC  ,iC1,iC2]-1]

            eq_p = self.ss[iC+1].EQs[index_p,iC1, iC2]
            eq   = self.ss[iC  ].EQs[index,iC1, iC2]
            
            return p* self.Phi(eq_p.vN1,eq_p.vI1) + (1-p) *self.Phi(eq.vN1,eq.vI1) 
        
        def H2(iC1, iC2, iC):
            index_p  = self.ess.esr[self.ess.index[iC+1,iC1,iC2]-1]
            index    = self.ess.esr[self.ess.index[iC  ,iC1,iC2]-1]

            eq_p = self.ss[iC+1].EQs[index_p,iC1, iC2]
            eq   = self.ss[iC  ].EQs[index,iC1, iC2]
            return p* self.Phi(eq_p.vN2,eq_p.vI2) + (1-p) *self.Phi(eq.vN2,eq.vI2) 

        
        a = self.r1( self.C[ic1],self.C[ic2] ) - self.K(c) + self.beta * H1(ic,ic2,ic)
        b = self.beta * (   H1(ic,ic,ic) - H1(ic,ic2,ic)  ) 
        
        eq_p = self.ss[ic+1].EQs[h,ic1,ic2]
        
        d = self.r1( self.C[ic1],self.C[ic2] )   + self.beta * p * self.Phi(eq_p.vN1 , eq_p.vI1  ) 
        e = self.beta * H1(ic1,ic,ic)         - self.beta * p * self.Phi( eq_p.vN1 , eq_p.vI1  ) 

        pa = - self.beta * (1-p) * b
        pb = e + ( self.beta * (1-p) -1) * b - self.beta * (1-p) * a
        pc = d + ( self.beta * (1-p) -1 ) * a

        # Solve for p2 mixed strategy ... but also returns 1 and 0 for pure
        pstar2 = self.quad(pa,pb,pc)

        A = self.r2(self.C[ic1],self.C[ic2]) - self.K(c) + self.beta * H2(ic1,ic,ic)
        B = self.beta * ( H2(ic,ic,ic) - H2(ic1,ic,ic) )
        D = self.r2(self.C[ic1],self.C[ic2]) + self.beta * p * self.Phi( eq_p.vN2 , eq_p.vI2 )
        E = self.beta * H2(ic,ic2,ic)       - self.beta * p * self.Phi( eq_p.vN2 , eq_p.vI2 )

        qa = - self.beta * (1-p) * B
        qb = E + ( self.beta * (1-p) - 1 ) * B - self.beta * (1-p) * A
        qc = D + ( self.beta * (1-p) - 1 ) * A

        pstar1 = self.quad(qa, qb, qc)


        count = 0


        for i in range(len(pstar1)):
            for j  in range(len(pstar2)):
                if np.all( [k in [0,1] for k in [i,j] ] ): # these are pure strategies
                    # If the polynomial is negative vI > vN
                    # hence player invests set exPj=1 else 0
                    # exP1 is best response to pstar2(j)
                    exP1 = pc + pb * pstar2[j] + pa * pstar2[j]**2 < 0 
                    exP2 = qc + qb * pstar1[i] + qa * pstar1[i]**2 < 0 

                    # check if both are playing best response
                    # in pure strategies. Players best response
                    # should be equal to the candidate to which
                    # the other player is best responding.
                    if np.abs(exP1 - pstar1[i]) < 1e-7 and np.abs(exP2-pstar2[j]) < 1e-7:
                        # if exP1=0 and pstar_i=0 true
                        # if exP1=1 and pstar_i=1 true
                        # Testing whether best response exP1 is
                        # equal to pstar1(i) to which Player 2
                        # is best responding ...
                        count = count + 1;
                        vI1 = a + b*pstar2[j] 
                        vN1 = (d + e*pstar2[j] + self.beta*q*(1-pstar2[j])*(a+b*pstar2[j]))*pstar1[i]     +     (1-pstar1[i])*(d+e*pstar2[j])/(1-self.beta*q*(1-pstar2[j]))
                        vI2 = A + B*pstar1[i]
                        vN2 = (D + E*pstar1[i] + self.beta*q*(1-pstar1[i])*(A+B*pstar1[i]))*pstar2[j]     +     (1-pstar2[j])*(D+E*pstar1[i])/(1-self.beta*q*(1-pstar1[i]))

                        #vN2 = (D +             + self.beta*q*(1          )*(A            ))*pstar2[j]     +     
                        
                        '''
                        debugging:
                        if ic1==0 and ic2==0:
                            for name, value in zip(['vN2','A','B','D','E','q','pstar1','pstar2'],[vN2,A,B,D,E,q,pstar1[i],pstar2[j]]):
                                print(name)
                                print(value)
                        '''

                        self.ss[ic].EQs[count-1, ic1, ic2]   = SimpleNamespace(P1=pstar1[i],vN1=vN1,vI1= vI1,P2=pstar2[j],vN2=vN2,vI2= vI2)

                elif i > 1 and j > 1 and pstar1[i] >= 0 and pstar2[j] >= 0 and pstar1[i] <= 1 and pstar2[j] <= 1:
                    count = count + 1
                    v1 = a + b * pstar2[j]
                    v2 = A + B * pstar1[i]
                    self.ss[ic].EQs[count-1, ic1, ic2] = SimpleNamespace(P1=pstar1[i],vN1=v1,vI1= v1,P2=pstar2[j],vN2=v2,vI2= v2)

        self.ss[ic].nEQ[ic1,ic2] = count
        self.ess.bases[self.ess.index[ic,ic1,ic2] -1]= count


######### printing and plotting 
    def print_eq(self,print_x = ['P1','vN1','vI1','P2','vN2','vI2'], layer=4, n_eq=1):
        # print equilibrum for a single layer and equilbrium
        for x in print_x:
            vec = np.empty((layer,layer))
            vec[:,:] =np.nan
            for i in range(layer):
                for j in range(layer):
                    get = self.ss[layer-1].EQs[n_eq-1,i,j]
                    if hasattr(get,x):
                        vec[i,j] = getattr(get,x)

            print(f'\n{x}:')
            print(vec)


    def print_equilibria(self,ESS,TAU,out,a = 10,d = 0.002):
        number_of_equilibria=np.size(TAU)

        print('\n')
        print(f'{number_of_equilibria} equilibria found.')
        T = np.size(self.stage_index)

        y = np.zeros((T,),dtype=int);

        for i in range(T):
            y[i] = np.sum( TAU==i+1)

        
        plt.bar(np.array((range(len(y))) ) +1,y )
        plt.title('Recursion start, frequencies')
        plt.show()

        print('STAGE       Recursion_started_in_stage :\n')
        for i, x in enumerate(y):
            print(f'{i+1:2}            {x:10}')





        V = np.empty((number_of_equilibria,2))
        MPEesr  = np.empty((number_of_equilibria,self.stage_index[-1].astype(int)))

        for iEQ in range(number_of_equilibria):
            V[iEQ,0]=out[iEQ].V1;
            V[iEQ,1]=out[iEQ].V2;
            MPEesr[iEQ,:]=out[iEQ].MPEesr; 
        

        
        
        df = pd.DataFrame(np.around(V, decimals=3)).rename(columns={0:'V1',1:'V2'})
        vc = pd.DataFrame(df.value_counts()).reset_index()
        vc['weight']= a + vc[0]/(d*np.max(vc[0]))
        vc.plot.scatter(x='V1',y='V2',s='weight',figsize=(8,8) , title='Distrubution of value functions in equilibrium');