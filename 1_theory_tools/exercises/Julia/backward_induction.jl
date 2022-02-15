
struct Par
    beta::Float64
    W::Int64
    T::Int64
end


function solve_backwards(par)
    
    Vstar_bi = Array{Float64}(undef, par.W+1,par.T)
    Cstar_bi = Array{Int64}(undef, par.W+1,par.T)
    
    # Initialize last period
    Cstar_bi[:,par.T] = Array(0:par.W);
    Vstar_bi[:,par.T] = sqrt.(Cstar_bi[:,par.T] )
    
    # Recursively solve earlier periods: 
    for t = par.T-1:-1:1
        for w = 0:par.W
            c = Array(0:w)
            V_next = Vstar_bi[w.-c.+1,t+1]
            v_guess = sqrt.(c)+par.beta.*V_next    

            star = findmax(v_guess)
            Vstar_bi[w+1,t] = star[1] 
            Cstar_bi[w+1,t] = star[2]-1

        end
    end
    return Cstar_bi,Cstar_bi
end
    


module vfi

mutable struct Sol
    grid_w::Array
    C_star::Array
    V_star::Array
    delta::Float64
end

function solve(par)

    
    #Initiate solution
    sol = Sol(Array(0:par.W),
            zeros(Int64,par.W+1),
            zeros(Float64,par.W+1),
            par.tol*100  
            )
    it =0
    
    
    while (par.max_iter>=it) & (par.tol<sol.delta)
        it = it+1
        V_next = copy(sol.V_star)
        

        for w in sol.grid_w
            c=Array(0:w)
           
            
            V_vec = sqrt.(c).+par.beta .*V_next[w.-c.+1]
            

            star = findmax(V_vec)
            
            sol.V_star[w+1] = star[1] 
            sol.C_star[w+1] = star[2]-1
        
        sol.delta = findmax(abs.(sol.V_star .- V_next))[1]
            
        end
    

    end
    
return sol
end

end 