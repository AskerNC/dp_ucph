
import numpy as np

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
    

        for key,value in dict_.items():
            setattr(self,key,value)

        # update dependent parameters 
        self.c = np.linspace(self.maxc,0,self.N)
        self.prob = np.ones((1,self.N-1)) * self.pf 

        # Functional forms

    def k(self,c):
        return self.k0+self.k1/(1+c)
    def payoff(self,x):
        return (self.p-x) * self.dt

        '''#Social planners productions cost
	    x = np.min(self.c)
	    # State of the art production cost
        c = x
	    # Reservation price
	    p = self.p
	    # Discount rate
	    v_N = self.payoff(x)/(1-self.beta) #Value of not investing
	    v_I = v_N - self.K(c) # Value of investing
	    V = np.max(v_I,v_N) #  Value of value function in the state (x_K,c_K)
	    policy = v_I > v_N # Chosen policy = 1 if investing'''


    def solve_last_corner(self):
        x  = np.min(self.c)
        c = x
        v_N = self.payoff(x)/(1-self.beta) #Value of not investing
        v_I = v_N - self.K(c)
        V=np.max(v_I,v_N)
        policy = (v_I >v_N)

        return V,policy
        

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


        # Start by solving the last corner
        V[-1,-1] , P[-1,-1] = self.solve_last_corner()

        for ix in range(self.N-2,-1,-1):
            pass