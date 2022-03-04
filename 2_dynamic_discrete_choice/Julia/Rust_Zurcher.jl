
# Using https://mark-ponder.com/tutorials/discrete-choice-models/dynamic-discrete-choice-nested-fixed-point-algorithm/

module Rust
using Optim

mutable struct Model
    beta::Float64 # Discount Factor
    params::Array{Float64} # Utility Function Parameters
    pi::Array{Float64} # Transition Probabilities
    EV::Array{Float64} # Expected Value Array
    K::Int32 # Size of State Space
 
   # Initialize Model Parameters
    function Model( beta = .9999, K = 90; 
                        pi = [.348, .639, .013],params = [3.6, 10])
       EV = ones(K, 1)   
       
       return new( beta, params, pi, EV, K )
    end
end;


mutable struct Data
    endog::Array{Int32}
    exog::Array{Int32}
 
    function Data( endog = [], exog = [])
        return new( endog, exog )
    end
end


function u(model::Model)
    S = collect(1:model.K)' # Generate State Space range, i.e. [1, 2, 3, 4, ...]
    d_0 = .001 * model.params[1] * S # Utility from not replacing
    d_1 = model.params[2] * ones(1, model.K) # Utility from replacing
    U = vcat(d_0, d_1) # Utility matrix
    return -U
end



end