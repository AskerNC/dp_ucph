
module Deaton

using Parameters # For setting default values in struct with @with_kw
using UnPack # For unpacking 
using FastGaussQuadrature
using Distributions
using Interpolations
using ..Struct_types
#include("Struct_types.jl"jupy)

"""Class to implement deaton's model with log-normally distrubuted income shocks"""

    @with_kw mutable struct deaton_model <: Struct_types.econmodel
        
        # structural parameters
        β::Float64 =0.9     # Discount factor
        R::Float64 =1.       # Returns on savings
        μ::Float64=0.        # Location parameter for income shock, y (if y is log normal, ln(y) ~ N(μ, σ^2))  
        σ::Float64=1.       # Scale parameter for income shock, y
        η::Float64=1.       # CRRA utility parameter, (η=0: linear, η=1: log, η>1: more risk averse than log

        #spaces 
        xbar::Array=[2e-16, 10.] # Upper and lower bound on cash on hand
        n_x::Int64=50       # Number of grid points for cash on hand
        n_c::Int64=100      # Number of grid points for choice grid
        n_y::Int64=10       # Number of quadrature points for income
        integration::String = "Hermite"


        # Spaces dependent on size
        x::Array = Array{Float64}(LinRange(xbar[1],xbar[2],n_x))
        c::Array = initiate_c(n_x,n_c,x,xbar)

        quad::NamedTuple = inititate_integration(n_y,μ,σ , integration)

    end


    function initiate_c(n_x::Int64,n_c::Int64,x::Array,xbar::Array)
        c = Array{Float64}(undef,(n_x,n_c))
        
        for i in 1:1:n_x
            c[i,:] = LinRange(xbar[1],x[i],n_c)
        end
        return c
    end


    function inititate_integration(n_y::Int64, μ::Float64, σ::Float64, integration::String )
        
        if integration =="Hermite"
            q, w = gausshermite(n_y)
            
            y = exp.(μ .+σ .* sqrt(2) .* q)
            
            weights = w ./ sqrt(pi)
            
        elseif integration == "Legendre"
            q, w = gausslegendre(n_y)
            weights = w ./ 2
            q_ = (q .+ 1 ) ./2
            y = exp.(quantile.(Normal(μ,σ), q_) )
        else
            error("Non-supported integration type")      
        end
        
            
        return (y = y ,weights =weights)
    end



    function u(dm::deaton_model,c::Array)
        if dm.η ==1
            return log.(c)
        elseif dm.η>=0
            return (c.^(1-dm.η).-1)./(1 .-dm.η)
        end 

    end


    function bellman(dm::deaton_model,V0::Array)
        "Bellman operator, V0 is one-dim vector of values on state grid"
        interp = LinearInterpolation(dm.x,V0,extrapolation_bc=Periodic())
        EV=zeros(Float64,dm.n_x,dm.n_c)

        for (i, y_i) in enumerate(dm.quad.y)
            x1 = dm.R .* (dm.x .- dm.c ) .+ y_i
            
            EV .+= dm.quad.weights[i] .* interp.(x1)
        end
        
        matV1 = u(dm,dm.c) .+ dm.β .*EV
        
        star = findmax(matV1,dims=2)

        V1 = vec(star[1])
        c1 = dm.c[star[2]]



        return V1, c1
    end

end
