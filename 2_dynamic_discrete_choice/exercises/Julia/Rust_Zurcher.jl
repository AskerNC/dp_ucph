
# Using https://mark-ponder.com/tutorials/discrete-choice-models/dynamic-discrete-choice-nested-fixed-point-algorithm/

module Rust


    using Optim, SparseArrays, DataFrames, LinearAlgebra
  
    mutable struct Solution
        optimum
        params::Array
        se::Array

    # Initialize Model Parameters
        function Solution( optimum="",params=[],se=[])
               
        return new(optimum, params,se)
        end
    end



    mutable struct Model
        β::Float64 # Discount Factor
        params::Array{Float64} # Utility Function Parameters
        π::Array{Float64} # Transition Probabilities
        EV::Array{Float64} # Expected Value Array
        K::Int64 # Size of State Space
        max::Int64 # Max of mileage
        sol:: Solution
        P1:: SparseMatrixCSC{Float64, Int64}
        P2:: SparseMatrixCSC{Float64, Int64}
    
    # Initialize Model Parameters
        function Model(;    β = .9999, K = 90,
                            π = [.348, .639, .013],params = [3.6, 10], max= 450)
        EV = zeros(K, 1)   
        sol=Solution()
        P1, P2 = transition_probs((π=π, K=K); both =true)

        
        return new( β, params, π , EV, K, max,sol,P1,P2)
        end
    end


    mutable struct Data
        endog::Array{Int32}
        exog::Array{Int32}
    
        function Data( df::DataFrame,model::Model)
            I = ( df[:,:id] - [0; df[1:end-1,:id]  ] .==0) 
            endog= df[I,:d]
            exog= ceil.(Int64,df[I,:x] * model.K / (model.max*1000)) 
            
            return new( endog, exog )
        end
    end


    function u(model::Model)
        S = collect(0:model.K-1)' # Generate State Space range, i.e. [1, 2, 3, 4, ...] but -1 instead
        d_0 = .001 * model.params[1] * S # Utility from not replacing
        d_1 = model.params[2] * ones(1, model.K) # Utility from replacing
        U = vcat(d_0, d_1) # Utility matrix
        return -U
    end
    
    
    function transition_probs(model; both=false)
        t = length(model.π)
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
            
            atmp[coor:coor+t-i-1] = [model.π[1:t - i - 1]' ( 1 - sum(model.π[1:t- i - 1]) ) ]
            
            x = model.K-t+1+i
            x_coor[r+coor:r+coor+t-i-1] = repeat([x],t-i)
            y_coor[r+coor:r+coor+t-i-1] = [x+l for l in 0:t-i-1]

            coor += t-i
        end
        
        values =  [repeat(model.π,model.K-t+1); atmp]
        
        
        P1  =sparse( x_coor, y_coor, values)


        if both
            P2 = spzeros(model.K,model.K)
            for i in 1:length(model.π)
                P2[:,i] .= model.π[i] 
            end

            return P1, P2  
        end

        return P1
    end
    


    """
    #Original way which does not utilize sparse matrices
    function transition_probs(model::Model)
        t = length(model.π)
        ttmp = zeros(model.K - t, model.K) # Transition Probabilities
        for i in 1:model.K - t
            for j in 0:t-1
                ttmp[i, i + j] = model.π[j+1]
            end
        end
        atmp = zeros(t,t) # Absorbing State Probabilities
        for i in 0:t - 1
            atmp[i+ 1,:] = [zeros(1,i) model.π[1:t - i - 1]' ( 1 - sum(model.π[1:t- i - 1]) ) ]
        end
        return [ttmp ; zeros(t, model.K - t) atmp]
    end;
    """



    function bellman(model::Model;output=1)
        u = Rust.u(model)
        value_keep      = u[1,:] + model.β *model.EV
        value_replace   = u[2,:] .+ model.β *model.EV[1] 
        ss_val = (  exp.( value_keep      - model.EV  ) +
                    exp.( value_replace  .-  model.EV ) )

        logsum = model.EV + log.(ss_val)
        ev = model.P1 * logsum
        if output ==1
            return ev
        end

        # Choice probality
        pk = 1 ./ (1 .+ exp.(value_replace-value_keep) )    
        if output == 2
            return ev, pk
        end 

        
        dev1 = dbellman(model,pk)

        return ev1, pk, dev1
    end


    function dbellman(model::Model,pk::Array)
        dev1 = model.β .* model.P1 .* transpose(pk)
        dev1[:,1] += model.β .*  model.P1 * (1 .- pk)
        return dev1
    end



    function ss(model::Model)
        u = Rust.u(model)
        ss_val = (  exp.( u[1,:] + model.β *model.EV      - model.EV  ) +
                    exp.( u[2,:] .+ model.β *model.EV[1]   .-  model.EV ) )

        return model.EV + log.(ss_val)
    end


    function contraction_mapping( model::Model )
        #P = sparse( transition_probs( model ) ) # Transition Matrix (K x K)
        P = Rust.transition_probs( model ) # Transition Matrix (K x K)
        eps = 1 # Set epsilon to something greater than 0
        while eps > .000001
            EV1 = P * Rust.ss( model )
            eps = maximum(abs.(EV1 - model.EV))
            model.EV = EV1
        end
    end

    function choice_p(model::Model)
        max_EV = maximum(model.EV)
        u= Rust.u(model)

        P_k = exp.( u[1,:] + model.β *model.EV .-max_EV) ./  (  
                        exp.( u[1,:] + model.β *model.EV       .- max_EV  ) +
                        exp.( u[2,:] .+ model.β *model.EV[1]   .- max_EV) )
        return P_k 
    end


    function partialLL( model::Model, data::Data)
        decision_obs = data.endog
        state_obs = data.exog
        cp_tmp = Rust.choice_p( model )
        relevant_probs = [ cp_tmp[convert(Int, i)] for i in state_obs ]
        pll = [ if decision == 0 log(r_p) else log(1 - r_p) end for (decision, r_p) in zip(decision_obs, relevant_probs)]
        return -sum(pll)
    end
     
    function ll( model::Model, data::Data )
     
        function objFunc( params )
            model.params = params
            Rust.contraction_mapping( model )
            pll = Rust.partialLL( model, data )
            return pll
        end
     
        params0 = copy(model.params)
        optimum = optimize( objFunc, params0 )

        model.sol.optimum = optimum
        model.sol.params = model.sol.optimum.minimizer
        return optimum
    end


    function inference(model::Model,data::Data)
        model.params = model.sol.params

        tmpP = Rust.choice_p(model)
        byOb = tmpP[data.exog,:]

        T = length(data.exog)
        tp = Rust.transition_probs(model) 

        tmpT2 = tp[:,2:end] .* tmpP[2:end,:]'
        
        
        dR = -(1 .- tp * tmpP)
        dTheta = - (tp*(1:1:90)*-0.001) .* tmpP
        dEV = inv(Array((I -  hcat(1 .- sum(tmpT2,dims=2) , tmpT2) .* model.β))) * hcat(dTheta, dR)

        # Derivative of utility with respect to parameters
        
        tmp = -(1 .- byOb .* (data.endog .== 0) .- (1 .-byOb) .* (data.endog .== 1))


        score = hcat( -.001*data.exog, -ones(T, 1)).*tmp

        # Add the derivative of the difference in Expected Value
        score .+= model.β*(dEV[1,:]' .* ones(T,1) .- dEV[data.exog, :]) .*tmp
        
        # Calculate inverse of the covariance
        H = inv(score'score)
        model.sol.se = sqrt.(diag(H))

        hcat(model.sol.params, model.sol.se)

    end





end