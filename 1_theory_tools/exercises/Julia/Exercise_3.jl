

module ex3
using Interpolations
using Optim
using Parameters # For setting default values in struct with @with_kw
using UnPack # For unpacking 

@with_kw mutable struct Sol
    W_grid::Array
    C::Array
    V::Array
    delta::Float64 = 100000.
    it::Int64 = 0
end

function solve_consumption_grid_search(par::NamedTuple)
    
    
    sol = Sol(W_grid = par.W_grid,
            C= zeros(Float64,par.num_W),
            V= zeros(Float64,par.num_W),
            )
    
    while (par.max_iter>= sol.it) & (par.tol<sol.delta)
        sol.it +=1
        V_next = copy(sol.V)

        
        interp = LinearInterpolation(par.W_grid,V_next)

        for (iw,w) in enumerate(par.W_grid)  # enumerate automaticcaly unpack w
            
            
            fun(x) = -V(x,w,interp,par)       
            
            res = optimize(fun,1e-8,1-1e-8)
            
            sol.V[iw] = -res.minimum
            sol.C[iw] = res.minimizer*w
            

        end        
        sol.delta = findmax(abs.(sol.V .- V_next))[1]
    end 
    
    return sol
end


function V(x::Float64,w::Float64,interp::Interpolations.Extrapolation,par::NamedTuple)
    c = x*w

    wt1 = w-c

    return sqrt(c)+par.β *interp(wt1)
end

function solve_consumption_grid_search2(par::NamedTuple)

    
    sol = Sol(W_grid = par.W_grid,
            C= zeros(Float64,par.num_W),
            V= zeros(Float64,par.num_W),
            )
    
    grid_C = LinRange(0, 1.0, par.num_C)

    
    while (par.max_iter>= sol.it) & (par.tol<sol.delta)
        sol.it +=1
        V_next = copy(sol.V)

        
        interp = LinearInterpolation(par.W_grid,V_next)

        for (iw,w) in enumerate(par.W_grid)  # enumerate automaticcaly unpack w
            c = grid_C.*w
            wt1 = w .- c
            V_guess = sqrt.(c) .+ par.β .* interp.(wt1)
            
            star = findmax(V_guess)
            
            sol.V[iw] = star[1] 
            sol.C[iw] = (star[2]-1)*w
                        

        end        
        sol.delta = findmax(abs.(sol.V .- V_next))[1]
    end 
    
    return sol
end

end