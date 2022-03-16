module Chebyshev

Tn(x,n) = cos(n*acos(x))


function interpolation(fhandle,points::AbstractArray,m::Int64, n::Int64)
    # This is the Chebyshev Interpolation (Regression algorithm)      
    #  in approximation of a scalar function, f(x):R->R                
    #    The approach follow Judd (1998, Allgortihm 6.2, p. 223)     

    #############################################################################
# INPUT ARGUMENTS:
#             fhandle:               The funtion, that should be approximated
#             points:              The interval for the approximation of f(x).
#             m:                     number of nodes used to construct the approximation. NOTE: m>=n+1
#             n:                     Degree of approximation-polynomial
# 
# OUTPUT ARGUMENTS:
#             f_approx:              The vector of approximated function values
#             f_actual:              The vector of actual function values
#             points:                The vector of points, for which the function is approximated
##################################################################################



    @assert (m>=n+1) "The specified parameters are not acceptable. Make sure m>n"
    
    a = points[1]
    b = points[end]

    number = size(points)
    f_approx = Array{Float64}(undef,number)
    f_actual = Array{Float64}(undef,number)


    for x in 1:number[1]
        ai = Array{Float64}(undef,n+1)
        f_hat = 0
        for i in 0:n
            
            nom =0 
            denom = 0 
            for k in 0:m-1

                zk = -cos(((2*(k+1)-1)/(2*m))*pi)

                xk = (zk+1)*((b-a)/2)+a


                yk = fhandle(xk)

                nom += yk* Tn(zk,i)
                denom += Tn(zk,i)^2

                if k==m-1
                    
                    ai[i+1] = nom/denom
                end
            end
                        
            f_hat = f_hat + ai[i+1]*Tn(2*(points[x]-a)/(b-a)-1,i)
    
        end 


        f_approx[x] = f_hat
        f_actual[x] = fhandle(points[x]) 
        
    end

    return f_approx, f_actual, points

end


end 
