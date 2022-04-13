
class Ap():

    def __init__(self,dict_={}) :
            
        self.printfxp =2
        self.sa_max=200
        self.sa_min = 10
        self.sa_tol = 1e-10
        self.maxfpiter =5
        self.pi_max = 90
        self.pi_tol = 1e-12
        self.tol_ratoi = 1e-12


        for key,value in dict_.items():
            setattr(self,key,value)
    