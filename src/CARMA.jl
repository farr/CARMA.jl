module CARMA

include("Celerite.jl")
include("Kepler.jl")
include("CARMAKepler.jl")
include("Kalman.jl")
include("Green.jl")

using .Kepler
using .CARMAKepler
using .Celerite
using .Green
using .Kalman

export Green, Kalman, Celerite, CARMAKepler, Kepler

end
