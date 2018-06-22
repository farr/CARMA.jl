# Improves performance on my machine.
BLAS.set_num_threads(1)

using Ensemble
using CARMA
using HDF5

usage = "julia run_carma.jl DATAFILE P Q [NLIVE]"

if size(ARGS, 1) < 3
    println(usage)
    exit(1)
end

datafile = ARGS[1]
p = parse(Int, ARGS[2])
q = parse(Int, ARGS[3])

nlive = 1024
if size(ARGS, 1) == 4
    nlive = int(ARGS[4])
end

nmcmc = 32

outfile = "state-$(p)-$(q).hdf5"
ckpt_file = "state-$(p)-$(q).ckpt"

data = readdlm(datafile)
data = data[sortperm(data[:,1]),:]

ts = Float64[data[1,1]]
ys = Float64[data[1,2]]
dys = Float64[data[1,3]]

for i in 2:size(data,1)
    t = data[i,1]
    y = data[i,2]
    dy = data[i,3]

    if t == ts[end]
        dy2 = dy*dy
        dys2 = dys[end]*dys[end]

        yy = (y*dys2 + ys[end]*dy2)/(dy2 + dys2)
        dyy = dy*dys[end]/sqrt(dys2 + dy2)

        ys[end] = yy
        dys[end] = dyy
    else
        push!(ts, t)
        push!(ys, y)
        push!(dys, dy)
    end
end

# Fix any zeros in dy---set to minimum positive dy
dys[dys.<=0] = minimum(dys[dys.>0])

post = Kalman.CARMAKalmanPosterior(ts, ys, dys, p, q)

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

EnsembleNest.run!(nest_state, 0.01, ckpt_file=ckpt_file)

h5open(outfile, "w") do f
    ng = g_create(f, "nest_state")
    write(ng, nest_state)

    pg = g_create(f, "carma_data")
    pg["ts", "compress", 3, "shuffle", ()] = ts
    pg["ys", "compress", 3, "shuffle", ()] = ys
    pg["dys", "compress", 3, "shuffle", ()] = dys
    pg["p"] = p
    pg["q"] = q
end

if ispath(ckpt_file)
    rm(ckpt_file)
end
