

module backwards
mutable struct Sol
    W_grid::Array
    C::Array
    V::Array
end

function solve(par)
    
    sol =   Sol(Array(0:par.W),
            Array{Int64}(undef, par.W+1,par.T),
            Array{Float64}(undef, par.W+1,par.T)
            )

    
    # Initialize last period
    sol.C[:,par.T] = sol.W_grid;
    sol.V[:,par.T] = sqrt.(sol.C[:,par.T] )
    
    # Recursively solve earlier periods: 
    for t = par.T-1:-1:1
        for w in sol.W_grid
            c = Array(0:w)
            V_next = sol.V[w.-c.+1,t+1]
            v_guess = sqrt.(c)+par.beta.*V_next    

            star = findmax(v_guess)
            sol.V[w+1,t] = star[1] 
            sol.C[w+1,t] = star[2]-1

        end
    end
    return sol
end

end


module vfi

mutable struct Sol
    W_grid::Array
    C::Array
    V::Array
    delta::Float64
    it::Int64
end

function solve(par)

    
    #Initiate solution
    sol = Sol(Array(0:par.W),
            zeros(Int64,par.W+1),
            zeros(Float64,par.W+1),
            par.tol*100,
            0
            )
    
    
    
    while (par.max_iter>=sol.it) & (par.tol<sol.delta)
        sol.it = sol.it+1
        V_next = copy(sol.V)
        

        for w in sol.W_grid
            c=Array(0:w)
           
            
            V_vec = sqrt.(c).+par.beta .*V_next[w.-c.+1]
            

            star = findmax(V_vec)
            
            sol.V[w+1] = star[1] 
            sol.C[w+1] = star[2]-1
        
        sol.delta = findmax(abs.(sol.V .- V_next))[1]
            
        end
    

    end
    
return sol
end


mutable struct Sim
    W::Int64
    T::Int64
    C::Array
end

function simulate(sol,T,W)
    sim = Sim(W,T,Array{Int64}(undef,T))

    W_now = sim.W
    for t = 1:T
        sim.C[t]=sol.C[W_now+1]
        W_now -= sim.C[t]
    end

    return sim
end


end 