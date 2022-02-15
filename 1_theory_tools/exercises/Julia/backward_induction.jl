
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

struct Par
    beta::Float64
    W::Int64
    max_iter::Int64
    delta::Int64
    tol::Float64
end

mutable struct Sol
    grid_w::Array
    Cstar::Array
    V_star::Array
end

end