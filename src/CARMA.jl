module CARMA

include("Kalman.jl")
include("Green.jl")

using .Green
using .Kalman

export Green, Kalman

end 
