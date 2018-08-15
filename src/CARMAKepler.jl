module CARMAKepler

using ..Celerite
using Distributions
using Ensemble
using HDF5
using ..Kepler

import Base:
    write, read

struct MultiEpochPosterior
    ts::Array{Array{Float64, 1}, 1}
    ys::Array{Array{Float64, 1}, 1}
    dys::Array{Array{Float64, 1}, 1}

    inds::Array{Array{Int, 1}, 1}

    allts::Array{Float64, 1}
    allys::Array{Float64, 1}
    alldys::Array{Float64, 1}

    P_min::Float64
    P_max::Float64

    K_min::Float64
    K_max::Float64

    ndrw::Int
    nosc::Int

    rate_min::Float64
    rate_max::Float64

    f_min::Float64
    f_max::Float64

    rms_min::Float64
    rms_max::Float64

    Q_min::Float64
    Q_max::Float64
end

function MultiEpochPosterior(ts, ys, dys, per_min, per_max, ndrw, nosc, rate_min, rate_max, f_min, f_max, Q_max)
    allts = vcat(ts...)
    allys = vcat(ys...)
    alldys = vcat(dys...)

    allinds = sortperm(allts)

    iallinds = invperm(allinds)

    inds = []
    i = 1
    for t in ts
        push!(inds, iallinds[i:i+size(t,1)-1])
        i = i+size(t,1)
    end

    allts = allts[allinds]
    allys = allys[allinds]
    alldys = alldys[allinds]

    rms_max = maximum([std(y) for y in ys])
    rms_min = minimum([std(y) for y in ys])

    T = allts[end]-allts[1]
    dtmin = minimum(diff(allts))

    MultiEpochPosterior(ts, ys, dys, inds, allts, allys, alldys, per_min, per_max, rms_min/100.0, 10.0*rms_max, ndrw, nosc, rate_min, rate_max, f_min, f_max, rms_min/100.0, rms_max*10.0, 1.0, Q_max)
end

struct MultiEpochParams
    mu::Array{Float64, 1}
    nu::Array{Float64, 1}

    K::Float64
    P::Float64
    e::Float64
    omega::Float64
    chi::Float64

    drw_rms::Array{Float64,1}
    drw_rate::Array{Float64,1}

    osc_rms::Array{Float64, 1}
    osc_freq::Array{Float64, 1}
    osc_Q::Array{Float64, 1}
end

function nparams(post::MultiEpochPosterior)
    2*size(post.ts, 1) + 5 + 2*post.ndrw + 3*post.nosc
end

function to_params(post::MultiEpochPosterior, x::Array{Float64, 1})
    i = 1
    mu = zeros(size(post.ts, 1))
    nu = zeros(size(post.ts, 1))
    for j in 1:size(post.ts, 1)
        mu[j] = x[i]
        nu[j] = Parameterizations.bounded_value(x[i+1], 0.5, 2.0)
        i = i + 2
    end

    K = Parameterizations.bounded_value(x[i], post.K_min, post.K_max)
    P = Parameterizations.bounded_value(x[i+1], post.P_min, post.P_max)
    ecosw, esinw = Parameterizations.unit_disk_value(x[i+2:i+3])
    e = sqrt(ecosw*ecosw + esinw*esinw)
    omega = atan(esinw, ecosw)
    chi = Parameterizations.bounded_value(x[i+4], 0.0, 1.0)
    i = i + 5

    drw_rms = zeros(post.ndrw)
    drw_rate = zeros(post.ndrw)

    rmin = post.rate_min
    for j in 1:post.ndrw
        drw_rms[j] = Parameterizations.bounded_value(x[i], post.rms_min, post.rms_max)
        drw_rate[j] = Parameterizations.bounded_value(x[i+1], rmin, post.rate_max)
        i = i + 2
        rmin = drw_rate[j]
    end

    osc_rms = zeros(post.nosc)
    osc_freq = zeros(post.nosc)
    osc_Q = zeros(post.nosc)
    freq_min = post.f_min
    for j in 1:post.nosc
        osc_rms[j] = Parameterizations.bounded_value(x[i], post.rms_min, post.rms_max)
        osc_freq[j] = Parameterizations.bounded_value(x[i+1], freq_min, post.f_max)
        osc_Q[j] = Parameterizations.bounded_value(x[i+2], post.Q_min, post.Q_max)
        i = i + 3
        freq_min = osc_freq[j]
    end

    MultiEpochParams(mu, nu, K, P, e, omega, chi, drw_rms, drw_rate, osc_rms, osc_freq, osc_Q)
end

function to_array(post::MultiEpochPosterior, p::MultiEpochParams)
    x = zeros(nparams(post))

    i = 1
    for j in 1:size(post.ts,1)
        x[i] = p.mu[j]
        x[i+1] = Parameterizations.bounded_param(p.nu[j], 0.5, 2.0)
        i = i + 2
    end

    x[i] = Parameterizations.bounded_param(p.K, post.K_min, post.K_max)
    x[i+1] = Parameterizations.bounded_param(p.P, post.P_min, post.P_max)
    x[i+2:i+3] = Parameterizations.unit_disk_param([p.e*cos(p.omega), p.e*sin(p.omega)])
    x[i+4] = Parameterizations.bounded_param(p.chi, 0.0, 1.0)
    i = i + 5

    rmin = post.rate_min
    for j in 1:post.ndrw
        x[i] = Parameterizations.bounded_param(p.drw_rms[j], post.rms_min, post.rms_max)
        x[i+1] = Parameterizations.bounded_param(p.drw_rate[j], rmin, post.rate_max)
        i = i + 2
        rmin = p.drw_rate[j]
    end

    freq_min = post.f_min
    for j in 1:post.nosc
        x[i] = Parameterizations.bounded_param(p.osc_rms[j], post.rms_min, post.rms_max)
        x[i+1] = Parameterizations.bounded_param(p.osc_freq[j], freq_min, post.f_max)
        x[i+2] = Parameterizations.bounded_param(p.osc_Q[j], post.Q_min, post.Q_max)
        i = i + 3
        freq_min = p.osc_freq[j]
    end

    x
end

function log_prior(post::MultiEpochPosterior, x::Array{Float64, 1})
    log_prior(post, to_params(post, x))
end

function bounded_logjac_value(x, low, high)
    Parameterizations.bounded_logjac(x, Parameterizations.bounded_param(x, low, high), low, high)
end

function log_prior(post::MultiEpochPosterior, p::MultiEpochParams)
    logp = 0.0

    for i in 1:size(post.ts, 1)
        mu = mean(post.ys[i])
        sigma = std(post.ys[i])

        logp += logpdf(Normal(mu, 10*sigma), p.mu[i])

        logp -= log(p.nu[i]) # flat in log(nu)
        logp += bounded_logjac_value(p.nu[i], 0.5, 2)
    end

    logp -= log(p.K)
    logp += bounded_logjac_value(p.K, post.K_min, post.K_max)

    logp -= log(p.P)
    logp += bounded_logjac_value(p.P, post.P_min, post.P_max)

    # Uniform prior on unit disk for (e*cos(omega), e*sin(omega))
    z = [p.e*cos(p.omega), p.e*sin(p.omega)]
    logp += Parameterizations.unit_disk_logjac(z, Parameterizations.unit_disk_param(z))

    # Uniform prior in chi
    logp += bounded_logjac_value(p.chi, 0.0, 1.0)

    rmin = post.rate_min
    for i in 1:post.ndrw
        logp -= log(p.drw_rms[i])
        logp += bounded_logjac_value(p.drw_rms[i], post.rms_min, post.rms_max)

        logp -= log(p.drw_rate[i])
        logp += bounded_logjac_value(p.drw_rate[i], rmin, post.rate_max)

        rmin = p.drw_rate[i]
    end

    fmin = post.f_min
    for i in 1:post.nosc
        logp -= log(p.osc_rms[i])
        logp += bounded_logjac_value(p.osc_rms[i], post.rms_min, post.rms_max)

        logp -= log(p.osc_freq[i])
        logp += bounded_logjac_value(p.osc_freq[i], fmin, post.f_max)
        fmin = p.osc_freq[i]

        logp -= log(p.osc_Q[i])
        logp += bounded_logjac_value(p.osc_Q[i], post.Q_min, post.Q_max)
    end

    logp
end

function draw_prior(post::MultiEpochPosterior, n)
    hcat([draw_prior(post) for i in 1:n]...)
end

function rand_flatlog(low, high)
    exp(log(low) + rand()*log(high/low))
end

function draw_prior(post::MultiEpochPosterior)
    nts = size(post.ts, 1)

    mus = zeros(nts)
    nus = zeros(nts)

    for i in eachindex(mus)
        mu = mean(post.ys[i])
        sigma = std(post.ys[i])

        mus[i] = mu + 10*sigma*randn()

        nus[i] = rand_flatlog(0.5, 2.0)
    end

    K = rand_flatlog(post.K_min, post.K_max)
    P = rand_flatlog(post.P_min, post.P_max)

    x = 1.0
    y = 1.0
    while x*x + y*y > 1.0
        x = rand()
        y = rand()
    end

    e = sqrt(x*x + y*y)
    omega = atan(y,x)

    chi = rand()

    drw_rmss = zeros(post.ndrw)
    drw_rates = zeros(post.ndrw)

    for i in eachindex(drw_rmss)
        drw_rmss[i] = rand_flatlog(post.rms_min, post.rms_max)
        drw_rates[i] = rand_flatlog(post.rate_min, post.rate_max)
    end
    inds = sortperm(drw_rates)
    drw_rmss = drw_rmss[inds]
    drw_rates = drw_rates[inds]

    osc_rmss = zeros(post.nosc)
    osc_freqs = zeros(post.nosc)
    osc_Qs = zeros(post.nosc)

    for i in eachindex(osc_rmss)
        osc_rmss[i] = rand_flatlog(post.rms_min, post.rms_max)
        osc_freqs[i] = rand_flatlog(post.f_min, post.f_max)
        osc_Qs[i] = rand_flatlog(post.Q_min, post.Q_max)
    end
    inds = sortperm(osc_freqs)
    osc_rmss = osc_rmss[inds]
    osc_freqs = osc_freqs[inds]
    osc_Qs = osc_Qs[inds]

    p = MultiEpochParams(mus, nus, K, P, e, omega, chi, drw_rmss, drw_rates, osc_rmss, osc_freqs, osc_Qs)

    to_array(post, p)
end

function log_likelihood(post::MultiEpochPosterior, x::Array{Float64, 1})
    log_likelihood(post, to_params(post, x))
end

function produce_ys_dys(post::MultiEpochPosterior, p::MultiEpochParams)
    n = size(post.allts, 1)

    ys = zeros(n)
    dys = zeros(n)

    for i in eachindex(post.ts)
        ys[post.inds[i]] = post.ys[i] .- p.mu[i]
        dys[post.inds[i]] = post.dys[i].*p.nu[i]
    end

    for i in eachindex(ys)
        ys[i] = ys[i] - Kepler.rv(post.allts[i], p.K, p.P, p.e, p.omega, p.chi)
    end

    ys, dys
end

function log_likelihood(post::MultiEpochPosterior, p::MultiEpochParams)
    ys, dys = produce_ys_dys(post, p)

    filt = Celerite.CeleriteKalmanFilter(0.0, p.drw_rms, p.drw_rate, p.osc_rms, p.osc_freq, p.osc_Q)

    Celerite.log_likelihood(filt, post.allts, ys, dys)
end

function residuals(post::MultiEpochPosterior, x::Array{Float64, 1})
    residuals(post, to_params(post, x))
end

function residuals(post::MultiEpochPosterior, p::MultiEpochParams)
    ys, dys = produce_ys_dys(post, p)

    filt = Celerite.CeleriteKalmanFilter(0.0, p.drw_rms, p.drw_rate, p.osc_rms, p.osc_freq, p.osc_Q)

    Celerite.residuals(filt, post.allts, ys, dys)
end

function psd(post::MultiEpochPosterior, x::Array{Float64, 1}, fs::Array{Float64, 1})
    psd(post, to_params(post, x), fs)
end

function psd(post::MultiEpochPosterior, p::MultiEpochParams, fs::Array{Float64, 1})
    filt = Celerite.CeleriteKalmanFilter(0.0, p.drw_rms, p.drw_rate, p.osc_rms, p.osc_freq, p.osc_Q)

    Celerite.psd(filt, fs)
end

function predict(post::MultiEpochPosterior, x::Array{Float64, 1}, ts::Array{Float64, 1})
    predict(post, to_params(post, x), ts)
end

function predict(post::MultiEpochPosterior, p::MultiEpochParams, ts::Array{Float64, 1})
    rvs = [Kepler.rv(t, p.K, p.P, p.e, p.omega, p.chi) for t in ts]

    ys, dys = produce_ys_dys(post, p)

    filt = Celerite.CeleriteKalmanFilter(0.0, p.drw_rms, p.drw_rate, p.osc_rms, p.osc_freq, p.osc_Q)

    ysp, vysp = Celerite.predict(filt, post.allts, ys, dys, ts)

    ysp = ysp .+ rvs

    ysp, vysp
end

function posterior_predictive(post::MultiEpochPosterior, x::Array{Float64, 1})
    posterior_predictive(post, to_params(post, x))
end

function posterior_predictive(post::MultiEpochPosterior, p::MultiEpochParams)
    filt = Celerite.CeleriteKalmanFilter(0.0, p.drw_rms, p.drw_rate, p.osc_rms, p.osc_freq, p.osc_Q)

    allnoise = Celerite.generate(filt, post.allts, zeros(size(post.allts, 1)))

    allrvnoise = zeros(size(allnoise, 1))
    for i in eachindex(allnoise)
        allrvnoise[i] = allnoise[i] + Kepler.rv(post.allts[i], p.K, p.P, p.e, p.omega, p.chi)
    end

    obs_ys = Array{Float64, 1}[]
    for i in eachindex(post.ts)
        ys = allrvnoise[post.inds[i]]
        push!(obs_ys, ys + p.nu[i]*randn(size(ys, 1)) + p.mu[i])
    end

    obs_ys
end

function write(f::Union{HDF5File, HDF5Group}, post::MultiEpochPosterior, ns::EnsembleNest.NestState)
    ns_group = g_create(f, "nest_state")
    write(ns_group, ns)

    xs, lnlikes = EnsembleNest.postsample(ns)

    ps = [to_params(post, xs[:,j]) for j in 1:size(xs,2)]

    f["lnlike", "compress", 3, "shuffle", ()] = lnlikes
    f["mu", "compress", 3, "shuffle", ()] = hcat([p.mu for p in ps]...)
    f["nu", "compress", 3, "shuffle", ()] = hcat([p.nu for p in ps]...)
    f["K", "compress", 3, "shuffle", ()] = [p.K for p in ps]
    f["P", "compress", 3, "shuffle", ()] = [p.P for p in ps]
    f["e", "compress", 3, "shuffle", ()] = [p.e for p in ps]
    f["omega", "compress", 3, "shuffle", ()] = [p.omega for p in ps]
    f["chi", "compress", 3, "shuffle", ()] = [p.chi for p in ps]

    f["drw_rms", "compress", 3, "shuffle", ()] = hcat([p.drw_rms for p in ps]...)
    f["drw_rate", "compress", 3, "shuffle", ()] = hcat([p.drw_rate for p in ps]...)

    f["osc_rms", "compress", 3, "shuffle", ()] = hcat([p.osc_rms for p in ps]...)
    f["osc_freq", "compress", 3, "shuffle", ()] = hcat([p.osc_freq for p in ps]...)
    f["osc_Q", "compress", 3, "shuffle", ()] = hcat([p.osc_Q for p in ps]...)
end

function read(f::Union{HDF5File, HDF5Group}, post::MultiEpochPosterior)
    mus = read(f, "mu")
    nus = read(f, "nu")
    Ks = read(f, "K")
    Ps = read(f, "P")
    es = read(f, "e")
    omegas = read(f, "omega")
    chis = read(f, "chi")
    drw_rmss = read(f, "drw_rms")
    drw_rates = read(f, "drw_rate")
    osc_rmss = read(f, "osc_rms")
    osc_freqs = read(f, "osc_freq")
    osc_Qs = read(f, "osc_Q")

    ps = [MultiEpochParams(mus[:,i], nus[:,i], Ks[i], Ps[i], es[i], omegas[i], chis[i], drw_rmss[:,i], drw_rates[:,i], osc_rmss[:,i], osc_freqs[:,i], osc_Qs[:,i]) for i in eachindex(Ks)]

    logl = x -> log_likelihood(post, x)
    logp = x -> log_prior(post, x)

    ns = EnsembleNest.NestState(f["nest_state"], logl=logl, logp=logp)

    (ps, ns)
end

end
