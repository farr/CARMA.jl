"""
Kalman filters, log-likelihood and reasonable priors for a "Celerite"
process, as defined in [Foreman-Mackey, et
al. (2017)[https://arxiv.org/abs/1703.09710].

A Celerite process is a sum of AR(1) and CARMA(2,0) processes; an
alternative description is as a sum of damped random walks and SHOs
driven by white noise.  It is, in fact, a re-parameterization of a
CARMA process, but a convenient one for stably finding mode
frequencies and damping rates and correlation timescales because it is
a sum of low-order terms.

"""

module Celerite

using Ensemble

mutable struct CeleriteKalmanFilter
    mu::Float64 # Mean
    x::Array{ComplexF64, 1} # State mean
    Vx::Array{ComplexF64, 2} # State variance
    K::Array{ComplexF64, 1} # Kalman Gain
    lambda::Array{ComplexF64, 1} # Evolution factors exp(roots*dt)
    b::Array{ComplexF64, 2} # Rotated observation vector
    roots::Array{ComplexF64, 1} # Eigenvalues of the ODEs
    V::Array{ComplexF64, 2} # Stationary covariance
    Vtemp::Array{ComplexF64, 2} # Storage for matrix ops
    drw_rms::Array{Float64, 1}
    drw_rates::Array{Float64, 1}
    osc_rms::Array{Float64, 1}
    osc_freqs::Array{Float64, 1}
    osc_Qs::Array{Float64, 1}
end

function reset!(filt::CeleriteKalmanFilter)
    p = size(filt.x, 1)
    filt.x = zeros(ComplexF64, p)
    filt.Vx = copy(filt.V)
    filt.K = zeros(ComplexF64, p)
    filt
end

function osc_roots(osc_freqs::Array{Float64, 1}, osc_Qs::Array{Float64, 1})
    nosc = size(osc_freqs, 1)

    oscroots = zeros(ComplexF64, nosc)
    for i in 1:nosc
        omega0 = 2.0*pi*osc_freqs[i]
        alpha = omega0/(2.0*osc_Qs[i])
        oscroots[i] = -alpha + omega0*1im
    end
    oscroots
end

function CeleriteKalmanFilter(mu::Float64, drw_rms::Array{Float64, 1}, drw_rates::Array{Float64, 1}, osc_rms::Array{Float64, 1}, osc_freqs::Array{Float64, 1}, osc_Qs::Array{Float64, 1})
    ndrw = size(drw_rms,1)
    nosc = size(osc_rms,1)
    dim = ndrw + 2*nosc

    oscroots = osc_roots(osc_freqs, osc_Qs)

    roots = zeros(ComplexF64, dim)
    ii = 1
    for i in 1:ndrw
        roots[ii] = -drw_rates[i]
        ii+=1
    end

    for i in 1:nosc
        roots[ii] = oscroots[i]
        roots[ii+1] = conj(oscroots[i])
        ii += 2
    end

    b = zeros(ComplexF64, dim)

    iobs = 1
    for i in 1:ndrw
        b[iobs] = 1.0
        iobs += 1
    end

    for i in 1:nosc
        b[iobs] = 1.0
        iobs += 2 # skip the derivative term
    end

    b = b'

    U = zeros(ComplexF64, (dim,dim))

    ii = 1
    for i in 1:ndrw
        U[ii,ii] = 1.0
        ii += 1
    end

    for i in 1:nosc
        U[ii,ii] = 1.0
        U[ii, ii+1] = 1.0
        U[ii+1,ii] = oscroots[i]
        U[ii+1,ii+1] = conj(oscroots[i])
        ii += 2
    end

    b = b*U # Rotated observation vector

    e = zeros(ComplexF64, dim)
    ii = 1
    for i in 1:ndrw
        e[ii] = 1.0
        ii += 1
    end

    for i in 1:nosc
        e[ii+1] = 1.0
        ii += 2
    end

    J = U \ e

    V = zeros(ComplexF64, (dim,dim))

    ii = 1
    for i in 1:ndrw
        V[ii,ii] = -J[ii]*conj(J[ii])/(roots[ii] + conj(roots[ii]))
        ii += 1
    end

    for i in 1:nosc
        for j in 0:1
            for k in 0:1
                V[ii+j, ii+k] = -J[ii+j]*conj(J[ii+k])/(roots[ii+j] + conj(roots[ii+k]))
            end
        end
        ii += 2
    end

    ii = 1
    for i in 1:ndrw
        s2 = b[1,ii]*V[ii,ii]*conj(b[1,ii])
        s2 = s2[1]
        V[ii,ii] *= drw_rms[i]*drw_rms[i]/s2
        ii += 1
    end

    for i in 1:nosc
        s2 = b[:,ii:ii+1]*V[ii:ii+1,ii:ii+1]*b[:,ii:ii+1]'
        s2 = s2[1]
        V[ii:ii+1, ii:ii+1] *= osc_rms[i]*osc_rms[i]/s2
        ii += 2
    end

    CeleriteKalmanFilter(mu, zeros(ComplexF64, dim), V, zeros(ComplexF64, dim), zeros(ComplexF64, dim), b, roots, copy(V), zeros(ComplexF64, (dim,dim)), drw_rms, drw_rates, osc_rms, osc_freqs, osc_Qs)
end

@inbounds function advance!(filt::CeleriteKalmanFilter, dt::Float64)
    p = size(filt.x, 1)

    for i in 1:p
        x = filt.roots[i]*dt
        filt.lambda[i] = exp(x)
    end
    lam = filt.lambda

    for i in 1:p
        filt.x[i] = lam[i]*filt.x[i]
    end

    for j in 1:p
        for i in 1:p
            a::ComplexF64 = lam[i]*conj(lam[j])
            b::ComplexF64 = a*(filt.Vx[i,j] - filt.V[i,j])
            filt.Vx[i,j] = b + filt.V[i,j]
        end
    end

    filt
end

@inbounds function observe!(filt::CeleriteKalmanFilter, y::Float64, dy::Float64)
    p = size(filt.x, 1)

    ey, vy = predict(filt)
    vy += dy*dy

    for i in 1:p
        filt.K[i] = zero(filt.K[i])
        for j in 1:p
            filt.K[i] += filt.Vx[i,j]*conj(filt.b[1,j])/vy
        end
    end

    for i in 1:p
        filt.x[i] = filt.x[i] + (y - ey)*filt.K[i]
    end

    for j in 1:p
        for i in 1:p
            a::ComplexF64 = vy*filt.K[i]
            b::ComplexF64 = a*conj(filt.K[j])
            filt.Vx[i,j] = filt.Vx[i,j] - b
        end
    end

    filt
end

@inbounds function predict(filt::CeleriteKalmanFilter)
    p = size(filt.x,1)

    yp = filt.mu
    for i in 1:p
        yp += real(filt.b[1,i]*filt.x[i])
    end

    vyp = 0.0
    for i in 1:p
        for j in 1:p
            a::ComplexF64 = filt.b[1,i]*filt.Vx[i,j]
            b::ComplexF64 = a*conj(filt.b[1,j])
            vyp = vyp + real(b)
        end
    end

    yp, vyp
end

function whiten(filt::CeleriteKalmanFilter, ts, ys, dys)
    n = size(ts, 1)

    reset!(filt)
    zs = zeros(n)

    for i in 1:n
        yp, vyp = predict(filt)

        zs[i] = (ys[i] - yp)/sqrt(vyp + dys[i]*dys[i])

        observe!(filt, ys[i], dys[i])

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    zs
end

function draw_and_collapse!(filt::CeleriteKalmanFilter)
    nd = size(filt.x, 1)
    try
        for i in 1:nd
            filt.Vx[i,i] = real(filt.Vx[i,i]) # Fix a roundoff error problem?
        end
        L = ctranspose(chol(Hermitian(filt.Vx)))
        filt.x = filt.x + L*randn(nd)
        filt.Vx = zeros(ComplexF64, (nd, nd))
    catch e
        if isa(e, Base.LinearAlgebra.PosDefException)
            warn("Current variance matrix not pos. def.---may be roundoff problem in generation.")
            F = eigfact(filt.Vx)
            for i in eachindex(F[:values])
                l = real(F[:values][i])
                v = F[:vectors][:,i]

                if l < 0.0
                    l = 0.0
                end

                filt.x = filt.x + sqrt(l)*randn()*v
            end
            filt.Vx = zeros(ComplexF64, (nd, nd))
        else
            rethrow()
        end
    end
end

function generate(filt::CeleriteKalmanFilter, ts, dys)
    n = size(ts, 1)
    nd = size(filt.x, 1)

    ys = zeros(n)

    reset!(filt)

    for i in 1:n
        # Draw a new state
        draw_and_collapse!(filt)

        y, _ = predict(filt)

        ys[i] = y + dys[i]*randn()

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    ys
end

function log_likelihood(filt, ts, ys, dys)
    n = size(ts, 1)

    ll = -0.5*n*log(2.0*pi)

    reset!(filt)
    for i in 1:n
        yp, vyp = predict(filt)

        if vyp < 0.0
            warn("Kalman filter has gone unstable!")
            return -Inf
        end

        dy = ys[i] - yp
        vy = vyp + dys[i]*dys[i]

        ll -= 0.5*log(vy)
        ll -= 0.5*dy*dy/vy

        observe!(filt, ys[i], dys[i])

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    ll
end

function residuals(filt, ts, ys, dys)
    n = size(ts, 1)

    resid = zeros(n)
    dresid = zeros(n)

    reset!(filt)
    for i in 1:n
        yp, vyp = predict(filt)

        if vyp < 0.0
            warn("Kalman filter has gone unstable!")
            return resid, dresid
        end

        resid[i] = ys[i] - yp
        dresid[i] = sqrt(vyp + dys[i]*dys[i])

        observe!(filt, ys[i], dys[i])

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    resid, dresid
end

function raw_covariance(ts::Array{Float64, 1}, dys::Array{Float64, 1}, drw_rms::Array{Float64, 1}, drw_rates::Array{Float64, 1}, osc_rms::Array{Float64, 1}, osc_freqs::Array{Float64, 1}, osc_Qs::Array{Float64,1})
    N = size(ts, 1)
    ndrw = size(drw_rms,1)
    nosc = size(osc_rms,1)

    cov = zeros((N,N))
    dts = zeros((N,N))

    oscroots = osc_roots(osc_freqs, osc_Qs)

    for j in 1:N
        for i in 1:N
            dts[i,j] = abs(ts[i] - ts[j])
        end
    end

    for i in 1:ndrw
        cov += drw_rms[i]*drw_rms[i]*exp.(-drw_rates[i].*dts)
    end

    for i in 1:nosc
        A = 1.0 / (-4.0*real(oscroots[i])*(conj(oscroots[i]) - oscroots[i])*oscroots[i])
        B = 1.0 / (-4.0*real(oscroots[i])*(oscroots[i] - conj(oscroots[i]))*conj(oscroots[i]))

        s2 = osc_rms[i]*osc_rms[i] / (A+B)
        cov = cov .+ real.(s2.*(A.*exp.(oscroots[i].*dts) .+ B.*exp.(conj(oscroots[i]).*dts)))
    end

    for i in 1:N
        cov[i,i] += dys[i]*dys[i]
    end

    cov
end

function psd_drw(rms_amp, damp_rate, fs)
    4.0*damp_rate*rms_amp*rms_amp./abs2.(2.0*pi*1im.*fs .+ damp_rate)
end

function psd_osc(rms_amp, freq, Q, fs)
    r1 = osc_roots([freq], [Q])[1]
    r2 = conj(r1)
    norm = 1.0/real(2.0*r1*(r1-r2)*(r1+r2))

    rms_amp*rms_amp/norm./abs2.((2.0*pi*1im.*fs .- r1).*(2.0*pi*1im.*fs .- r2))
end

function psd(filt::CeleriteKalmanFilter, fs::Array{Float64, 1})
    Pfs = zeros(size(fs,1))

    for i in 1:size(filt.drw_rms,1)
        Pfs .= Pfs .+ psd_drw(filt.drw_rms[i], filt.drw_rates[i], fs)
    end

    for i in 1:size(filt.osc_rms,1)
        Pfs .= Pfs .+ psd_osc(filt.osc_rms[i], filt.osc_freqs[i], filt.osc_Qs[i], fs)
    end

    Pfs
end

function predict(filt::CeleriteKalmanFilter, ts, ys, dys, tsp)
    # The algorithm here is to run the filter twice over the union of
    # the data and prediction times.  The first run is "forward" in
    # time, and predicts at each time *before* observing the output at
    # data times, and the second run is "backward" in time, and
    # predicts at each time *after* observing the output at data
    # times.  Weighted-averaging of the two runs together gives the
    # full internal state prediction incorporating all the data at the
    # given times.

    allts = vcat(ts, tsp)
    obsflag = convert(Array{Bool, 1}, vcat(trues(size(ts, 1)), falses(size(tsp, 1))))
    inds = sortperm(allts)
    rinds = reverse(inds)

    yspforward = zeros(size(allts, 1))
    vyspforward = zeros(size(allts, 1))

    reset!(filt)
    for i in eachindex(allts)
        yspforward[inds[i]], vyspforward[inds[i]] = predict(filt)

        if obsflag[inds[i]]
            observe!(filt, ys[inds[i]], dys[inds[i]])
        end

        if i < size(allts, 1)
            advance!(filt, allts[inds[i+1]]-allts[inds[i]])
        end
    end

    yspbackward = zeros(size(allts, 1))
    vyspbackward = zeros(size(allts, 1))

    reset!(filt)
    for i in eachindex(allts)
        if obsflag[rinds[i]]
            observe!(filt, ys[rinds[i]], dys[rinds[i]])
        end

        yspbackward[rinds[i]], vyspbackward[rinds[i]] = predict(filt)

        if i < size(allts, 1)
            advance!(filt, allts[rinds[i]] - allts[rinds[i+1]])
        end
    end

    ysp = (yspforward.*vyspbackward .+ yspbackward.*vyspforward) ./ (vyspbackward .+ vyspforward)
    vysp = 1.0./(1.0./vyspforward .+ 1.0./vyspbackward)

    ysp[size(ts,1)+1:end], vysp[size(ts, 1)+1:end]
end

struct CeleriteKalmanPosterior
    ts::Array{Float64, 1}
    ys::Array{Float64, 1}
    dys::Array{Float64, 1}
    ndrw::Int
    nosc::Int
    mumin::Float64
    mumax::Float64
    fmin::Float64
    fmax::Float64
    ratemin::Float64
    ratemax::Float64
    Qmin::Float64
    Qmax::Float64
    drw_rms_min::Float64
    drw_rms_max::Float64
    osc_rms_min::Float64
    osc_rms_max::Float64
end

struct CeleriteKalmanParams
    mu::Float64
    nu::Float64
    drw_rms::Array{Float64, 1}
    drw_rates::Array{Float64, 1}
    osc_rms::Array{Float64, 1}
    osc_freqs::Array{Float64, 1}
    osc_Qs::Array{Float64, 1}
end

function ndim(post::CeleriteKalmanPosterior)
    2 + 2*post.ndrw + 3*post.nosc
end

function to_params(post::CeleriteKalmanPosterior, x::Array{Float64, 1})
    nd = post.ndrw
    no = post.nosc

    mu = Parameterizations.bounded_value(x[1], post.mumin, post.mumax)
    nu = Parameterizations.bounded_value(x[2], 0.1, 10.0)
    drw_rms = Parameterizations.bounded_value.(x[3:3+nd-1], post.drw_rms_min, post.drw_rms_max)
    drw_rates = Parameterizations.bounded_value.(x[3+nd:3+2*nd-1], post.ratemin, post.ratemax)
    osc_rms = Parameterizations.bounded_value.(x[3+2*nd:3+2*nd+no-1], post.osc_rms_min, post.osc_rms_max)
    osc_freqs = Parameterizations.bounded_value.(x[3+2*nd+no:3+2*nd+2*no-1], post.fmin, post.fmax)
    osc_Qs = Parameterizations.bounded_value.(x[3+2*nd+2*no:3+2*nd+3*no-1], post.Qmin, post.Qmax)

    CeleriteKalmanParams(mu, nu, drw_rms, drw_rates, osc_rms, osc_freqs, osc_Qs)
end

function to_array(post::CeleriteKalmanPosterior, p::CeleriteKalmanParams)
    x = zeros(ndim(post))

    nd = post.ndrw
    no = post.nosc

    x[1] = Parameterizations.bounded_param(p.mu, post.mumin, post.mumax)
    x[2] = Parameterizations.bounded_param(p.nu, 0.1, 10)
    x[3:3+nd-1] = Parameterizations.bounded_param.(p.drw_rms, post.drw_rms_min, post.drw_rms_max)
    x[3+nd:3+2*nd-1] = Parameterizations.bounded_param.(p.drw_rates, post.ratemin, post.ratemax)
    x[3+2*nd:3+2*nd+no-1] = Parameterizations.bounded_param.(p.osc_rms, post.osc_rms_min, post.osc_rms_max)
    x[3+2*nd+no:3+2*nd+2*no-1] = Parameterizations.bounded_param.(p.osc_freqs, post.fmin, post.fmax)
    x[3+2*nd+2*no:3+2*nd+3*no-1] = Parameterizations.bounded_param.(p.osc_Qs, post.Qmin, post.Qmax)

    x
end

function log_prior(post::CeleriteKalmanPosterior, x::Array{Float64, 1})
    log_prior(post, to_params(post, x), x)
end

function log_prior(post::CeleriteKalmanPosterior, p::CeleriteKalmanParams)
    log_prior(post, p, to_array(post, p))
end

function log_prior(post::CeleriteKalmanPosterior, p::CeleriteKalmanParams, x::Array{Float64, 1})
    nd = post.ndrw
    no = post.nosc

    lp = 0.0

    # Flat prior on mu
    lp += Parameterizations.bounded_logjac(p.mu, x[1], post.mumin, post.mumax)

    # Flat-in-log prior on nu
    lp += -log(p.nu) + Parameterizations.bounded_logjac(p.nu, x[2], 0.1, 10.0)

    # Flat in log prior on rms
    lp += -sum(log.(p.drw_rms)) + sum(Parameterizations.bounded_logjac(p.drw_rms, x[3:3+nd-1], post.drw_rms_min, post.drw_rms_max))

    # Flat in log prior on rates
    lp += -sum(log.(p.drw_rates)) + sum(Parameterizations.bounded_logjac(p.drw_rates, x[3+nd:3+2*nd-1], post.ratemin, post.ratemax))

    # Flat in log prior on osc rms
    lp += -sum(log.(p.osc_rms)) + sum(Parameterizations.bounded_logjac(p.osc_rms, x[3+2*nd:3+2*nd+no-1], post.osc_rms_min, post.osc_rms_max))

    # Flat in log prior on osc freqs
    lp += -sum(log.(p.osc_freqs)) + sum(Parameterizations.bounded_logjac(p.osc_freqs, x[3+2*nd+no:3+2*nd+2*no-1], post.fmin, post.fmax))

    # Flat in log prior on Q
    lp += -sum(log.(p.osc_Qs)) + sum(Parameterizations.bounded_logjac(p.osc_Qs, x[3+2*nd+2*no:3+2*nd+3*no-1], post.Qmin, post.Qmax))

    lp
end

function log_likelihood(post::CeleriteKalmanPosterior, x::Array{Float64, 1})
    log_likelihood(post, to_params(post, x))
end

function log_likelihood(post::CeleriteKalmanPosterior, p::CeleriteKalmanParams)
    filt = CeleriteKalmanFilter(p.mu, p.drw_rms, p.drw_rates, p.osc_rms, p.osc_freqs, p.osc_Qs)

    log_likelihood(filt, post.ts, post.ys, p.nu*post.dys)
end

function init(post::CeleriteKalmanPosterior, n::Int)
    nd = post.ndrw
    no = post.nosc

    xs = zeros(ndim(post), n)

    for i in 1:n
        mu = post.mumin + (post.mumax-post.mumin)*rand()
        nu = exp(log(0.1) + (log(10.0) - log(0.1))*rand())

        drw_rms = exp.(log(post.drw_rms_min) + log(post.drw_rms_max/post.drw_rms_min)*rand(nd))
        drw_rates = exp.(log(post.ratemin) + log(post.ratemax/post.ratemin)*rand(nd))

        osc_rms = exp.(log(post.osc_rms_min) + log(post.rms_osc_max/post.osc_rms_min)*rand(no))
        osc_freqs = exp.(log(post.fmin) + log(post.fmax/post.fmin)*rand(no))
        osc_Qs = exp.(log(post.Qmin) + log(post.Qmax/post.Qmin)*rand(no))

        p = CeleriteKalmanParams(mu, nu, drw_rms, drw_rates, osc_rms, osc_freqs, osc_Qs)

        xs[:,i] = to_array(post, p)
    end

    xs
end

end
