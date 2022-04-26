
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
def empty_class():
    class sim: pass
    return sim
class Ssp_model():

    def __init__(self,dict_={}) :
            
        self.maxc= 5
        self.p = self.maxc
        
        self.k0 = 0
        self.k1=8.3
        self.N=100
        self.beta=0.99
        self.pf = 1
        self.beta = np.exp(-0.05)
        self.dt = 1
        self.T = 1000
        self.Kform = 'smooth'
    

        for key,value in dict_.items():
            setattr(self,key,value)

        # update dependent parameters 
        self.c = np.linspace(self.maxc,0,self.N)
        self.prob = np.ones((self.N,1)) * self.pf 
        self.prob[-1] = 0 

        # Functional forms

    def K(self,c):
        if self.Kform=='smooth':
            return self.k0+self.k1/(1+c)
        elif self.Kform=='constant':
            return self.k1

    def payoff(self,x):
        return (self.p-x) * self.dt


    def solve_last_corner(self):
        #Social planners productions cost
        x  = np.min(self.c)
        # State of the art production cost
        c = x

        v_N = self.payoff(x)/(1-self.beta) #Value of not investing
        v_I = v_N - self.K(c)
        V=np.max((v_I,v_N)) #  Value of value function in the state (x_K,c_K)
        policy = (v_I >v_N)  # Chosen policy = 1 if investing

        return V,policy
        
        
    def solve_last_interior(self,ix,jc,V):
        
        
        x = self.c[ix]
        c = self.c[jc]

        
        v_I = self.payoff(x) - self.K(c) + self.beta * V[jc,jc]
        v_N = self.payoff(x)/(1-self.beta)

        V= np.max((v_I,v_N))
        policy= (v_I>v_N)
        return V, policy

    def solve_corner(self,ix,jc,V):
        x = self.c[ix]
        # At the corner x=c
        c =x 
        

        v_N = ( self.payoff(x)+ self.beta* (self.prob[jc]) * V[ix,jc+1]  ) /(1-self.beta*(1-self.prob[jc])) #Value of not investing
        v_I = v_N - self.K(c)
        V=np.max((v_I,v_N)) #  Value of value function in the state (x_K,c_K)
        policy = (v_I >v_N)  # Chosen policy = 1 if investing
        return V, policy

    def solve_interior(self,ix,jc,V):
        x = self.c[ix]
        
        c = self.c[jc]


        # Calculate expected value of value function V(s') = V(x',c') 
        # conditional on investing a=1 and not investing a=1
        EV_1 = (1-self.prob[jc] ) *V[jc,jc] + self.prob[jc] * V[jc,jc+1]

        v_N = ( self.payoff(x)+ self.beta* self.prob[jc] *  V[ix,jc+1] )/(1-self.beta*(1-self.prob[jc])) #Value of not investing

        v_I = self.payoff(x)- self.K(c) + self.beta*EV_1

        V=np.max((v_I,v_N)) #  Value of value function in the state (x_K,c_K)
        policy = (v_I >v_N)  # Chosen policy = 1 if investing
        
        return V, policy

    def state_recursion(self):
        '''
        Solve social planners problmem with state recursion
			%
			%  [V, P] = spp.state_recursion(mp)
			%
			%  INPUT:
			%			mp:				Model parameters. See spp.setup
			%
			%  OUTPUT:
			%     V:        N x N matrix. Value function of social planner
			%			P:				Policy function of social planner
			% 							Column j of V and P corressponds to state of the art productions cost c = mp.c(j)
   		% 							Row i of V and P corressponds to productions cost x = mp.c(j)
        '''

        V = np.empty((self.N,self.N))
        P = np.empty((self.N,self.N))
        V[:,:] = np.nan
        P[:,:] = np.nan
        

        # Start by solving the last corner
        V[-1,-1] , P[-1,-1] = self.solve_last_corner()

        # Solve interior of last layer
        for ix in range(self.N-2,-1,-1):
            V[ix,self.N-1], P[ix,self.N-1]  = self.solve_last_interior(ix,self.N-1,V)

        #Then solve corner and interior of the rest of the layers:
        for jc in range(self.N-2,-1,-1):
            V[jc,jc], P[jc, jc]  = self.solve_corner(jc,jc,V)


            for ix in range(jc-1,-1,-1):
                V[ix,jc], P[ix,jc]  = self.solve_interior(ix,jc,V)


        self.V = V
        self.P = P

        return V, P 

    def simulate(self):
        # initial
        

        ix = [1]
        ic = [1]
        I = []
        

        np.random.seed(2020)
        u = np.random.uniform(size=self.T+1)

        for t in range(self.T):
            
            I.extend([ int(self.P[ix[t],ic[t]] )])
            ix.extend([ I[t]*ic[t] + (1-I[t])*ix[t]])
            ic.extend( ic[t] + (u[t] <self.prob[ic[t]]) )

            if ix[t]==self.N-1:
                print(f'got here: t={t}, ix={ix[t]}')
                break

        self.sim = empty_class()
        self.sim.ix = ix
        self.sim.ic = ic
        self.sim.I = I
        self.sim.t = list(range(t+1))
        self.sim.x = self.c[self.sim.ix]
        self.sim.c = self.c[self.sim.ic]

    def plot_solution(self):
        if not hasattr(self,'V'):
            self.state_recursion()

        for y in ['P', 'V']:
            df = getattr(self,y)
            #[::-1,::-1].T
            

            mask = np.ones(df.shape, dtype=bool, )
            mask[np.triu_indices_from(mask)] = False
            #mask = mask[::-1,::-1].T

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            #cmap = sns.diverging_palette(np.min(df) ,np.max(df), as_cmap=True)

            ticks = ['' for i in range(df.shape[0])]

            for i in range(self.maxc+1):        
                j = int(np.floor(i/(self.maxc) *(df.shape[0]-1)))

                ticks[j] = i  
            ticks.reverse()


            ax.set_title(y)
            ax.set_xlabel('State of the art marginal cost, c')
            ax.set_ylabel('Current marginal costs, x')
            # Draw the heatmap with the mask and correct aspect ratio
            sns_plot = sns.heatmap(df, mask=mask, cmap=cm.jet, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5},xticklabels=ticks,yticklabels=ticks)