

module dpsolver
using Interpolations
#include("Deaton.jl")
#include("Struct_types.jl")
using ..Deaton
using ..Struct_types


    function vfi_T(em::Struct_types.econmodel,T::Int64=100, callback="")
        """
        Class to implement deaton's model with log-normally distrubuted income shocks
        """
        #@time
            V = zeros(Float64,em.n_x,T)
            c = zeros(Float64,em.n_x,T)

            # In last period, consume everyting 
            V[:,T], c[:,T] = Deaton.bellman(em,zeros(Float64,em.n_x))

            for t = T:-1:2
                
                V[:,t-1], c[:,t-1] = Deaton.bellman(em,V[:,t])
                if callback!=""
                    callback(t,em.x,V,c)
                end
            end    
            
        
        #end
        return V,c
    end


    function vfi(em::Struct_types.econmodel,maxiter::Int64=100,tol::Float64=1e-6,callback="")
    """Solves the model using VFI (successive approximations)"""
        
        #@time
        #initiate
        V0 = zeros(Float64,em.n_x)
        V1 = zeros(Float64,em.n_x)
        c1 = zeros(Float64,em.n_x)
        
        for iter in 1:maxiter
            
            V1, c1 = Deaton.bellman(em,V0)
            
            if callback!=""
                callback(t,em.x,V,c)
            end
            if findmax(abs.(V1-V0))[1]<tol
                println("Solution found")
                #break
            else
                V0 = copy(V1)
            end
        end

        #end
        return V1 , c1
    end

    function iterinfo(iter,model,V1,V0, c="")
        println("Iter = $iter , ||V1-V0|| = $(findmax(abs.(V1-V0))[1])")
    end


end