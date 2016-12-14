using Ensemble
using CARMA

BLAS.set_num_threads(1)

usage = "julia run_multi_carma.jl P Q [NLIVE] FILE1 FILE2 ..."

if size(ARGS, 1) < 4
    println(usage)
    exit(1)
end

p = parse(Int, ARGS[1])
q = parse(Int, ARGS[2])

nlive = 1024
nlive_given = false
try
    nlive = parse(Int, ARGS[3])
    nlive_given = true
catch
    # Do nothing
end

files = ARGS[3:end]
if nlive_given
    files = ARGS[4:end]
end

nmcmc = 128

ts = Array{Float64, 1}[]
ys = Array{Float64, 1}[]
dys = Array{Float64, 1}[]
for f in files
    data = readdlm(f)

    t = data[:,1]
    y = data[:,2]
    dy = data[:,3]

    # Sorted in time
    inds = sortperm(t)
    t = t[inds]
    y = y[inds]
    dy = dy[inds]

    # Get rid of NaNs
    sel = ~isnan(y)
    t = t[sel]
    y = y[sel]
    dy = dy[sel]

    # Set dy == 0 to minimum positive dy
    @assert any(dy .> 0)
    sel = dy .== 0
    dy[sel] = minimum(dy[~sel])
    
    push!(ts, t)
    push!(ys, y)
    push!(dys, dy)
end

post = Kalman.MultiSegmentCARMAKalmanPosterior(ts, ys, dys, p, q)

nest_state = EnsembleNest.NestState(x -> Kalman.log_likelihood(post, x), x -> Kalman.log_prior(post, x), Kalman.init(post, nlive), nmcmc)

EnsembleNest.run!(nest_state, 0.1)

open("state-$(p)-$(q).dat", "w") do stream
    serialize(stream, (post, nest_state))
end
