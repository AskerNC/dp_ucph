
# Using https://mark-ponder.com/tutorials/discrete-choice-models/dynamic-discrete-choice-nested-fixed-point-algorithm/

module Rust
using Optim, SparseArrays

mutable struct Model
    beta::Float64 # Discount Factor
    params::Array{Float64} # Utility Function Parameters
    pi::Array{Float64} # Transition Probabilities
    EV::Array{Float64} # Expected Value Array
    K::Int32 # Size of State Space
 
   # Initialize Model Parameters
    function Model( beta = .9999, K = 90; 
                        pi = [.348, .639, .012,0.01],params = [3.6, 10])
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

function transition_probs(model::Model)
    t = length(model.pi)
    r = (model.K-t+1)*t
    size =  r + ((t-1)*(t))÷ 2
    
    x_coor = Array{Float64}(undef,size)
    y_coor = Array{Float64}(undef,size)
    
    for i in 1:model.K-t+1
        coor = (i-1)*t+1
        x_coor[coor:coor+t-1]= repeat([i],t)
        y_coor[coor:coor+t-1]= [i+l for l in 0:t-1]
    end

    
    atmp = Array{Float64}(undef,((t-1)*(t))÷ 2) # Absorbing State Probabilities
    
    coor = 1
    for i in 1:t - 1
        
        atmp[coor:coor+t-i-1] = [model.pi[1:t - i - 1]' ( 1 - sum(model.pi[1:t- i - 1]) ) ]
        
        x = model.K-t+1+i
        x_coor[r+coor:r+coor+t-i-1] = repeat([x],t-i)
        y_coor[r+coor:r+coor+t-i-1] = [x+l for l in 0:t-i-1]

        coor += t-i
    end
    
    values =  [repeat(model.pi,model.K-t+1); atmp]
    
    return sparse( x_coor, y_coor, values)
end;



"""
Original way which does not utilize sparse matrices
function transition_probs(model::Model)
    t = length(model.pi)
    ttmp = zeros(model.K - t, model.K) # Transition Probabilities
    for i in 1:model.K - t
        for j in 0:t-1
            ttmp[i, i + j] = model.pi[j+1]
        end
    end
    atmp = zeros(t,t) # Absorbing State Probabilities
    for i in 0:t - 1
        atmp[i+ 1,:] = [zeros(1,i) model.pi[1:t - i - 1]' ( 1 - sum(model.pi[1:t- i - 1]) ) ]
    end
    return [ttmp ; zeros(t, model.K - t) atmp]
end;
"""


end