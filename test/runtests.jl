using CARMA
using Test

# Ideas for tests to implement Any Day Now (TM)
#
# 1. Check against full-matrix likelihood.
# 2. Check that init returns always within prior.
# 3. Check to_params/to_array are inverses.

@testset "CARMA Tests" begin
    include("TestCelerite.jl")
    include("TestKalman.jl")
    include("TestLowPass.jl")
end
