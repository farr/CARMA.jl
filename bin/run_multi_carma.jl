using Ensemble
using CARMA
using HDF5

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

outfile = "state-$(p)-$(q).hdf5"
ckpt_file = "state-$(p)-$(q).ckpt"

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
function logl(x)
    Kalman.log_likelihood(post, x)
end
function logp(x)
    Kalman.log_prior(post, x)
end

if ispath(ckpt_file)
    nest_state = h5open(ckpt_file, "r") do f
        EnsembleNest.NestState(f, logl=logl, logp=logp)
    end
else
    nest_state = EnsembleNest.NestState(logl, logp, Kalman.init(post, nlive), nmcmc)
end

EnsembleNest.run!(nest_state, 0.1, verbose=true, ckpt_file=ckpt_file)

h5open(outfile, "w") do f
    ng = g_create(f, "nest_state")
    write(ng, nest_state)

    pg = g_create(f, "carma_data")
    pg["p"] = p
    pg["q"] = q
    pg["n"] = size(ts, 1)
    for i in eachindex(ts)
        gi = g_create(pg, "$(i)")
        gi["ts", "compress", 3, "shuffle", ()] = ts[i]
        gi["ys", "compress", 3, "shuffle", ()] = ys[i]
        gi["dys", "compress", 3, "shuffle", ()] = dys[i]
    end
end

if ispath(ckpt_file)
    rm(ckpt_file)
end
