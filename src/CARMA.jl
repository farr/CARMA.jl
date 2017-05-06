module CARMA

include("Celerite.jl")
include("Kalman.jl")
include("Green.jl")

using .Celerite
using .Green
using .Kalman

export Green, Kalman, Celerite

end 
