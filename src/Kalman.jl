module Kalman

using Base.LinAlg: PosDefException

export AR1KalmanFilter, reset!, advance!, observe!, generate, whiten, log_likelihood

type AR1KalmanFilter
    ypred::Float64
    vypred::Float64
    mu::Float64
    var::Float64
    tau::Float64
end

type AR1KalmanPosterior
    ts::Array{Float64, 1}
    ys::Array{Float64, 1}
    dys::Array{Float64, 1}
    mu::Float64
    sigma::Float64
    dtmin::Float64
    T::Float64
end

function AR1KalmanPosterior(ts, ys, dys)
    AR1KalmanPosterior(ts, ys, dys, mean(ys), std(ys), minimum(diff(ts)), ts[end]-ts[1])
end

type AR1KalmanParams
    mu::Float64
    sigma::Float64
    nu::Float64
    tau::Float64
end

function to_params(post::AR1KalmanPosterior, x::Array{Float64, 1})
    AR1KalmanParams(x[1], x[2], x[3], x[4])
end

function to_array(post::AR1KalmanPosterior, p::AR1KalmanParams)
    Float64[p.mu, p.sigma, p.nu, p.tau]
end

function log_prior(post::AR1KalmanPosterior, x::Array{Float64, 1})
    log_prior(post, to_params(post, x))
end

function rand_between(a, b)
    a + (b-a)*rand()
end

function rand_between(a, b, n)
    a + (b-a)*rand(n)
end

function init(post::AR1KalmanPosterior, n)
    pts = zeros(4,n)

    mu = post.mu
    sig = post.sigma

    dtmin = post.dtmin
    T = post.T

    pts[1,:] = rand_between((mu-10*sig), (mu+10*sig), n)
    pts[2,:] = exp(rand_between(log(sig/10.0),log(sig*10.0), n))
    pts[3,:] = exp(rand_between(log(0.1), log(10.0), n))
    pts[4,:] = exp(rand_between(log(0.1*dtmin),log(10.0*T), n))

    pts
end

function log_prior(post::AR1KalmanPosterior, p::AR1KalmanParams)
    mu = post.mu
    sig = post.sigma

    dtmin = post.dtmin
    T = post.T

    if p.mu < mu - 10*sig || p.mu > mu + 10*sig
        return -Inf
    end

    if p.sigma < sig/10.0 || p.sigma > sig*10.0
        return -Inf
    end

    if p.nu < 0.1 || p.nu > 10.0
        return -Inf
    end

    if p.tau < dtmin/10.0 || p.tau > 10.0*T
        return -Inf
    end

    # Flat in log(sigma), log(tau), log(nu); Flat in mu
    -log(p.sigma) - log(p.tau) - log(p.nu)
end

function log_likelihood(post::AR1KalmanPosterior, x::Array{Float64,1})
    log_likelihood(post, to_params(post, x))
end

function log_likelihood(post::AR1KalmanPosterior, p::AR1KalmanParams)
    filt = make_filter(post, p)

    log_likelihood(filt, post.ts, post.ys, p.nu*post.dys)
end

function make_filter(post, x::Array{Float64, 1})
    make_filter(post, to_params(post, x))
end

function make_filter(post::AR1KalmanPosterior, p::AR1KalmanParams)
    AR1KalmanFilter(p.mu, p.sigma*p.sigma, p.tau)
end

""" Produce a Kalman filter for an AR(1) process with the given mean,
variance, and timescale. """
function AR1KalmanFilter(mu, var, tau)
    AR1KalmanFilter(mu, var, mu, var, tau)
end

""" Reset the given filter to its initial state.  """
function reset!(filt::AR1KalmanFilter)
    filt.ypred = filt.mu
    filt.vypred = filt.var
end

""" Advance the given filter by a step `dt`.  """
function advance!(filt::AR1KalmanFilter, dt::Float64)
    lfac = exp(-dt/filt.tau)

    filt.ypred = filt.mu + (filt.ypred - filt.mu)*lfac
    filt.vypred = lfac*(filt.vypred - filt.var)*lfac + filt.var
end

""" Incorporate the given observation with measurement uncertainty (1
s.d.) into the filter estimate. """
function observe!(filt::AR1KalmanFilter, y::Float64, dy::Float64)
    vy = dy*dy + filt.vypred

    gain = filt.vypred / vy

    filt.ypred = filt.ypred + (y - filt.ypred)*gain
    filt.vypred = filt.vypred - gain*gain*vy
end

function predict(filt::AR1KalmanFilter)
    filt.ypred, filt.vypred
end

""" Generate the GP represented by the given filter, sampled at the
given times with the given observational uncertainties.  """
function generate(filt::AR1KalmanFilter, ts::Array{Float64,1}, dys::Array{Float64,1})
    reset!(filt)
    
    n = size(ts, 1)
    ys = zeros(n)

    ytrue = sqrt(filt.var)*randn() + filt.mu
    ys[1] = ytrue + dys[1]*randn()
    
    for i in 2:n
        observe!(filt, ytrue, 0.0)
        advance!(filt, ts[i]-ts[i-1])
        ytrue = filt.ypred + sqrt(filt.vypred)*randn()
        ys[i] = ytrue + dys[i]*randn()
    end

    ys
end

""" Whiten the observations assuming they are drawn from the GP
represented by the given filter.  The outputs will be independent
N(0,1) distributed. """
function whiten(filt::AR1KalmanFilter, ts::Array{Float64,1}, ys::Array{Float64, 1}, dys::Array{Float64,1})
    n = size(ts, 1)

    xs = zeros(n)

    reset!(filt)

    xs[1] = (ys[1] - filt.ypred)/sqrt(filt.vypred + dys[1]*dys[1])
    for i in 2:n
        observe!(filt, ys[i-1], dys[i-1])
        advance!(filt, ts[i]-ts[i-1])
        xs[i] = (ys[i] - filt.ypred) / sqrt(filt.vypred + dys[i]*dys[i])
    end

    xs
end

square(x) = x*x

function log_likelihood_term(filt, y, dy)
    var = filt.vypred + dy*dy
    -0.5*log(2.0*pi) - 0.5*log(var) - 0.5*square(y-filt.ypred)/var
end

""" The log-likelihood function for the given data assuming that it
comes from the GP represented by the filter. """
function log_likelihood(filt::AR1KalmanFilter, ts::Array{Float64,1}, ys::Array{Float64,1}, dys::Array{Float64,1})
    n = size(ts, 1)

    ll = zero(ys[1])
    
    reset!(filt)

    ll += log_likelihood_term(filt, ys[1], dys[1])
    for i in 2:n
        observe!(filt, ys[i-1], dys[i-1])
        advance!(filt, ts[i]-ts[i-1])
        ll += log_likelihood_term(filt, ys[i], dys[i])
    end

    ll
end

function psd(filt::AR1KalmanFilter, fs::Array{Float64, 1})
    4.0*filt.tau ./ (1.0 + (2.0*pi*filt.tau*fs).^2)
end

type CARMAKalmanFilter
    mu::Float64
    sig::Float64
    x::Array{Complex128,1}
    vx::Array{Complex128, 2}
    K::Array{Complex128, 1}
    lambda::Array{Complex128, 1}
    vxtemp::Array{Complex128, 2}
    v::Array{Complex128, 2}
    arroots::Array{Complex128, 1}
    b::Array{Complex128, 2}
    tscale::Float64
end

function reset!(filt::CARMAKalmanFilter)
    p = size(filt.x, 1)

    filt.x = zeros(Complex128, p)
    filt.vx = copy(filt.v)
    filt.K = zeros(Complex128, p)

    filt
end

"""Construct a polynomial from the given roots.  Returns an array of
coefficients, `c`, represting the polynomial as `p(x) =
sum(c[i]*x^(i-1))`.

"""
function poly{T <: Number}(roots::Array{T,1})
    n = size(roots, 1) + 1

    if n == 1
        return T[one(T)]
    else
        poly = zeros(T, n)

        poly[2] = one(T)
        poly[1] = -roots[1]

        for i in 2:size(roots,1)
            r = roots[i]
            for j in n:-1:2
                poly[j] = poly[j-1] - r*poly[j]
            end
            poly[1] = -poly[1]*r
        end

        poly
    end
end

function polyeval{T <: Number}(roots::Array{T, 1}, x::T)
    p = one(x)

    for i in 1:size(roots,1)
        p = p*(x - roots[i])
    end

    p
end

function CARMAKalmanFilter(mu::Float64, sigma::Float64, arroots::Array{Complex128, 1}, maroots::Array{Complex128, 1})
    p = size(arroots, 1)
    q = size(maroots, 1)

    @assert q < p "q must be less than p: q = $q, p = $p"
    @assert all(real(arroots) .< 0.0) "AR roots must have negative real part: $arroots"
    @assert all(real(maroots) .< 0.0) "MA roots must have negative real part: $maroots"

    tscale = exp(-mean(log(abs(arroots)))) # Try to stabilise the variance computation

    arroots = arroots*tscale
    maroots = maroots*tscale
    
    beta = poly(maroots)
    beta /= beta[1]
    b = cat(1, beta, zeros(p-q-1))
    b = b'

    U = zeros(Complex128, (p,p))
    for j in 1:p
        for i in 1:p
            U[i,j] = arroots[j]^(i-1)
        end
    end

    # Rotated observation vector
    b = b*U

    e = zeros(Complex128, p)
    e[end] = one(Complex128)

    J = U \ e

    V = zeros(Complex128, (p,p))
    for j in 1:p
        for i in 1:p
            V[i,j] = -J[i]*conj(J[j])/(arroots[i] + conj(arroots[j]))
        end
    end

    s2 = sigma*sigma/(b*V*b')[1]
    V = V*s2

    sig = sqrt(real(s2))

    CARMAKalmanFilter(mu, sig, zeros(Complex128, p), V, zeros(Complex128, p), zeros(Complex128, p), zeros(Complex128, (p,p)), copy(V), copy(arroots), b, tscale)
end

function advance!(filt::CARMAKalmanFilter, dt::Float64)
    p = size(filt.x, 1)

    for i in 1:p
        x = filt.arroots[i]*dt/filt.tscale
        filt.lambda[i] = exp(x)
    end
    lam = filt.lambda
        
    for i in 1:p
        filt.x[i] = lam[i]*filt.x[i]
    end

    for j in 1:p
        for i in 1:p
            filt.vx[i,j] = lam[i]*conj(lam[j])*(filt.vx[i,j] - filt.v[i,j]) + filt.v[i,j]
        end
    end

    filt
end

function observe!(filt::CARMAKalmanFilter, y::Float64, dy::Float64)
    p = size(filt.x, 1)

    ey, vy = predict(filt)
    vy += dy*dy

    for i in 1:p
        filt.K[i] = zero(filt.K[i])
        for j in 1:p
            filt.K[i] += filt.vx[i,j]*conj(filt.b[j])/vy
        end
    end

    for i in 1:p
        filt.x[i] = filt.x[i] + (y - ey)*filt.K[i]
    end

    for j in 1:p
        for i in 1:p
            filt.vx[i,j] = filt.vx[i,j] - vy*filt.K[i]*conj(filt.K[j])
        end
    end

    filt
end

function predict(filt::CARMAKalmanFilter)
    p = size(filt.x,1)
    
    yp = filt.mu
    for i in 1:p
        yp += real(filt.b[1,i]*filt.x[i])
    end

    vyp = 0.0
    for i in 1:p
        for j in 1:p
            vyp += real(filt.b[1,i]*filt.vx[i,j]*conj(filt.b[1,j]))
        end
    end

    yp, vyp
end

function whiten(filt::CARMAKalmanFilter, ts, ys, dys)
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

function draw_and_collapse!(filt::CARMAKalmanFilter)
    nd = size(filt.x, 1)
    try
        for i in 1:nd
            filt.vx[i,i] = real(filt.vx[i,i]) # Fix a roundoff error problem?
        end
        L = ctranspose(chol(Hermitian(filt.vx)))
        filt.x = filt.x + L*randn(nd)
        filt.vx = zeros(Complex128, (nd, nd))
    catch e
        if isa(e, Base.LinAlg.PosDefException)
            F = eigfact(filt.vx)
            for i in eachindex(F[:values])
                l = real(F[:values][i])
                v = F[:vectors][:,i]

                if l < 0.0
                    l = 0.0
                end
                
                filt.x = filt.x + sqrt(l)*randn()*v
            end
            filt.vx = zeros(Complex128, (nd, nd))
        else
            rethrow()
        end
    end
end

function generate(filt::CARMAKalmanFilter, ts, dys)
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

function carmacovariance(ts, sigma, arroots, maroots)
    n = size(ts, 1)
    p = size(arroots, 1)
    q = size(maroots, 1)

    @assert q < p

    beta = poly(maroots)
    beta /= beta[1]

    cm = zeros((n,n))

    for i in 1:n
        for j in i:n
            for k in 1:p
                r = arroots[k]
                br = zero(beta[1])
                bmr = zero(beta[1])
                rprod = one(arroots[1])
                for l in 1:q+1
                    br += beta[l]*r^(l-1)
                    bmr += beta[l]*(-r)^(l-1)
                end
                for l in 1:p
                    if l != k
                        rprod *= (arroots[l] - r)*(conj(arroots[l]) + r)
                    end
                end

                cm[i,j] += real(br*bmr*exp(r*abs(ts[j]-ts[i]))/(-2.0*real(r)*rprod))
            end
            cm[j,i] = cm[i,j]
        end
    end

    sfact = sigma*sigma/cm[1,1]
    for i in 1:n
        for j in 1:n
            cm[i,j] *= sfact
        end
    end

    cm
end

function carmagenerate(ts, dys, mu, sigma, arroots, maroots)
    n = size(ts, 1)

    cm = carmacovariance(ts, sigma, arroots, maroots)
    for i in 1:n
        cm[i,i] += dys[i]*dys[i]
    end

    L = chol(cm, Val{:L})

    ys = mu + L*randn(n)

    ys
end

function raw_carma_log_likelihood(ts, ys, dys, mu, sigma, arroots, maroots)
    n = size(ts, 1)

    cm = carmacovariance(ts, sigma, arroots, maroots)
    for i in 1:n
        cm[i,i] += dys[i]*dys[i]
    end

    zs = ys - mu

    F = cholfact(cm)
    L = F[:L]

    logdet = 0.0
    for i in 1:n
        logdet += log(L[i,i])
    end

    -0.5*n*log(2*pi) - logdet - 0.5*dot(zs, F \ zs)
end

type CARMAKalmanPosterior
    ts::Array{Float64, 1}
    ys::Array{Float64, 1}
    dys::Array{Float64,1}
    p::Int
    q::Int
end

function nparams(post::CARMAKalmanPosterior)
    post.p + post.q + 3
end

type CARMAPosteriorParams
    mu::Float64
    sigma::Float64
    nu::Float64
    arroots::Array{Complex128, 1}
    maroots::Array{Complex128, 1}
end

function to_roots(x::Array{Float64, 1})
    n = size(x, 1)
    r = zeros(Complex128, n)

    if n == 0
        r    
    elseif n == 1
        r[1] = -exp(x[1])
        r
    else
        for i in 1:2:n-1
            logb = x[i]
            logc = x[i+1]

            b = exp(logb)
            c = exp(logc)

            d = b*b - 4*c
            if d < 0.0
                sd = sqrt(-d)
                r[i] = -0.5*(b + sd*1im)
                r[i+1] = -0.5*(b - sd*1im)
            else
                sd = sqrt(d)
                r[i] = -0.5*(b + sd)
                r[i+1] = -0.5*(b - sd)
            end
        end

        if n % 2 == 1
            r[end] = -exp(x[end])
        end

        r
    end
end

function to_rparams(x::Array{Complex128, 1})
    n = size(x, 1)
    rp = zeros(n)

    if n == 0
        rp
    elseif n == 1
        rp = log(-real(r[1]))
    else
        for i in 1:2:n-1
            r1 = x[i]
            r2 = x[i+1]

            b = real(-(r1 + r2))
            c = real(r1*r2)

            rp[i] = log(b)
            rp[i+1] = log(c)
        end

        if n % 2 == 1
            rp[end] = log(-real(x[end]))
        end

        sort_root_params(rp)
    end
end

function root_params_ordered(rp::Array{Float64, 1})
    n = size(rp, 1)

    if n == 0
        true
    elseif n == 1
        true
    else
        x = -Inf
        for i in 1:2:n-1
            if rp[i] > x
                x = rp[i]
            else
                return false
            end
        end
        true
    end
end

function sort_root_params(rp::Array{Float64, 1})
    n = size(rp, 1)

    if n == 0
        rp
    elseif n == 1
        rp
    else
        logbs = rp[1:2:n-1]
        logcs = rp[2:2:n]
        inds = sortperm(logbs)

        rps = zeros(n)
        for i in 1:2:n-1
            rps[i] = logbs[inds[div(i,2)+1]]
            rps[i+1] = logcs[inds[div(i,2)+1]]
        end

        if n % 2 == 1
            rps[end] = rp[end]
        end

        rps
    end
end

function to_params(post::CARMAKalmanPosterior, x::Array{Float64,1})
    @assert size(x,1)==nparams(post)

    CARMAPosteriorParams(x[1], x[2], x[3], to_roots(x[4:3+post.p]), to_roots(x[4+post.p:end]))
end

function to_array(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    x = zeros(nparams(post))

    x[1] = p.mu
    x[2] = p.sigma
    x[3] = p.nu
    x[4:3+post.p] = to_rparams(p.arroots)
    x[4+post.p:end] = to_rparams(p.maroots)

    x
end

function rmin_rmax(post::CARMAKalmanPosterior)
    dt = minimum(diff(post.ts))
    T = post.ts[end] - post.ts[1]

    min_r = 1.0/T
    max_r = 1.0/dt

    min_r, max_r
end

function log_prior(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    if !root_params_ordered(x[4:3+post.p]) || !root_params_ordered(x[4+post.p:end])
        return -Inf
    end
    log_prior(post, to_params(post, x))
end

function log_prior(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    min_r, max_r = rmin_rmax(post)
    
    mu = mean(post.ys)
    sig = std(post.ys)

    if x.mu < mu - 10*sig || x.mu > mu + 10*sig
        return -Inf
    end

    if x.sigma < sig / 10.0 || x.sigma > sig * 10.0
        return -Inf
    end

    if x.nu < 0.1 || x.nu > 10.0
        return -Inf
    end

    for i in 1:post.p
        r = x.arroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    for i in 1:post.q
        r = x.maroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    -log(x.sigma) - log(x.nu)
end

function make_filter(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    CARMAKalmanFilter(post, p)
end

function CARMAKalmanFilter(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    CARMAKalmanFilter(post, to_params(post, x))
end

function CARMAKalmanFilter(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    CARMAKalmanFilter(p.mu, p.sigma, p.arroots, p.maroots)
end

function generate(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    generate(post, to_params(post, x))
end

function generate(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    dy = post.dys*p.nu
    
    y = generate(filt, post.ts, dy)

    y, dy
end

function log_likelihood(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    log_likelihood(post, to_params(post, x))
end

function log_likelihood(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    filt = nothing
    try
        filt = CARMAKalmanFilter(post, x)
    catch e
        if isa(e, DomainError)
            warn("Could not construct Kalman filter for parameters $(x)")
            return -Inf
        else
            rethrow()
        end
    end

    dys = post.dys * x.nu

    log_likelihood(filt, post.ts, post.ys, dys)
end

function log_posterior(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    log_posterior(post, to_params(post, x))
end

function log_posterior(post::CARMAKalmanPosterior, x::CARMAPosteriorParams)
    lp = log_prior(post, x)

    if lp == -Inf
        lp
    else
        lp + log_likelihood(post, x)
    end
end

function randroots(rmin, rmax, n)
    p = zeros(n)

    bmin = 2*rmin
    bmax = 2*rmax

    cmin = rmin*rmin
    cmax = 2.0*rmax*rmax

    logbmin = log(bmin)
    logbmax = log(bmax)

    logcmin = log(cmin)
    logcmax = log(cmax)
    
    for i in 1:2:n-1
        logb = 0.0
        logc = 0.0
        while true
            logb = logbmin + (logbmax-logbmin)*rand()
            logc = logcmin + (logcmax-logcmin)*rand()

            rs = to_roots(Float64[logb, logc])
            
            r1 = rs[1]
            r2 = rs[2]

            if real(r1) < -rmin && real(r1) > -rmax && imag(r1) > -rmax && imag(r1) < rmax && real(r2) < -rmin && real(r2) > -rmax && imag(r2) > -rmax && imag(r2) < rmax
                break
            end
        end

        p[i] = logb
        p[i+1] = logc
    end

    if n % 2 == 1
        p[end] = log(rmin) + (log(rmax)-log(rmin))*rand()
    end

    to_roots(p)
end

function init(post, n)
    rmin, rmax = rmin_rmax(post)

    mu0 = mean(post.ys)
    sig0 = std(post.ys)
    
    xs = zeros((nparams(post), n))

    for i in 1:n
        mu = mu0-10*sig0 + 20*sig0*rand()
        sig = exp(log(0.1*sig0) + rand()*(log(10.0*sig0) - log(0.1*sig0)))
        nu = exp(log(0.1) + rand()*(log(10.0)-log(0.1)))

        arroots = randroots(rmin, rmax, post.p)
        maroots = randroots(rmin, rmax, post.q)

        p = CARMAPosteriorParams(mu, sig, nu, arroots, maroots)
        xs[:,i] = to_array(post, p)
    end

    xs
end

function whiten(post::CARMAKalmanPosterior, p::Array{Float64, 1})
    whiten(post, to_params(post, p))
end

function whiten(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    whiten(filt, post.ts, post.ys, post.dys*p.nu)
end

""" Returns `(r, dr)` the residual and standard deviation on the
residual for the given parameters.
"""
function residuals(post::CARMAKalmanPosterior, p::Array{Float64, 1})
    residuals(post, to_params(post, p))
end

function residuals(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    residuals(filt, post.ts, post.ys, post.dys*p.nu)
end

function residuals(filt::CARMAKalmanFilter, ts::Array{Float64, 1}, ys::Array{Float64, 1}, dys::Array{Float64, 1})
    reset!(filt)

    rs = zeros(size(ts, 1))
    drs = zeros(size(ts, 1))

    for i in 1:size(ts, 1)
        y, vy = predict(filt)

        rs[i] = ys[i] - y
        drs[i] = sqrt(vy + dys[i]*dys[i])

        observe!(filt, ys[i], dys[i])

        if i < size(ts, 1)
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    rs, drs
end

function psdfreq(post::CARMAKalmanPosterior; nyquist_factor=1.0, oversample_factor=1.0)
    T = maximum(post.ts) - minimum(post.ts)
    dt_med = median(diff(post.ts))

    fmax = nyquist_factor/(2.0*dt_med)
    df = 1.0/(oversample_factor*T)

    collect(df:df:fmax)
end

""" Returns the one-sided PSD of the CARMA process in `post` described
by parameters `x` at the frequencies `fs`.  

The PSD is normalised so that ``\\int_{0}^{\\infty} df \\, P(f) =
\\sigma^2``, the variance of the process.  """
function psd(post::CARMAKalmanPosterior, x::Array{Float64, 2}, fs::Array{Float64, 1})
    psds = [psd(post, x[:,i], fs) for i in 1:size(x,2)]
    hcat(psds...)
end

function psd(post::CARMAKalmanPosterior, x::Array{Float64, 1}, fs::Array{Float64, 1})
    psd(post, to_params(post, x), fs)
end

function psd(post::CARMAKalmanPosterior, p::CARMAPosteriorParams, fs::Array{Float64, 1})
    psd = zeros(size(fs, 1))

    filt = CARMAKalmanFilter(post, p)
    
    for i in 1:size(fs, 1)
        f = fs[i]
        tpif = 2.0*pi*1.0im*f

        numer = polyeval(p.maroots, tpif) / polyeval(p.maroots, 0.0+0.0*1im)
        denom = polyeval(p.arroots, tpif)

        psd[i] = 2.0*filt.sig*filt.sig*abs2(numer)/abs2(denom) # 2.0 for one-sided PSD.
    end

    psd
end

function frequencies(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    frequencies(post, to_params(post, x))
end

function frequencies(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    imag(p.arroots)/(2.0*pi)
end

function frequencies(post::CARMAKalmanPosterior, x::Array{Float64, 2})
    fs = [frequencies(post, x[:,i]) for i in 1:size(x,2)]
    hcat(fs...)
end

function drates(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    drates(post, to_params(post, x))
end

function drates(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    real(p.arroots)
end

function drates(post::CARMAKalmanPosterior, x::Array{Float64, 2})
    drs = [drates(post, x[:,i]) for i in 1:size(x,2)]
    hcat(drs...)
end

function qfactors(post::CARMAKalmanPosterior, x::Array{Float64, 1})
    qfactors(post, to_params(post, x))
end

function qfactors(post::CARMAKalmanPosterior, p::CARMAPosteriorParams)
    fs = frequencies(post, p)
    dr = drates(post, p)

    pi*fs ./ dr
end

function qfactors(post::CARMAKalmanPosterior, x::Array{Float64, 2})
    qs = [qfactors(post, x[:,i]) for i in 1:size(x,2)]
    hcat(qs...)
end

""" Return the predicted state of the underlying process and the
variance in the prediction at the given times.  """
function predict(post::CARMAKalmanPosterior, x::Array{Float64, 1}, t::Float64)
    predict(post, to_params(post, x), t)
end

# This is now generic over the type of filt; should work for AR1 and CARMA.
function filter_prediction(filt, ts::Array{Float64, 1}, ys::Array{Float64, 1}, dys::Array{Float64, 1}, obs::Array{Bool, 1})
    n = size(ys, 1)

    yps = zeros(n)
    vyps = zeros(n)

    reset!(filt)

    for i in 1:n
        y,vy = predict(filt)

        yps[i] = y
        vyps[i] = vy

        if obs[i]
            observe!(filt, ys[i], dys[i])
        end

        if i < n
            advance!(filt, ts[i+1]-ts[i])
        end
    end

    yps, vyps
end

function predict(post, x::Array{Float64, 1}, ts::Array{Float64, 1})
    predict(post, to_params(post, x), ts)
end

function predict(post, p, ts::Array{Float64, 1})
    all_ts = vcat(post.ts, ts)
    all_obs = convert(Array{Bool, 1}, vcat(trues(size(post.ts,1)), falses(size(ts,1))))
    all_ys = vcat(post.ys, zeros(size(ts,1)))
    all_dys = vcat(post.dys, zeros(size(ts, 1)))

    iforward = sortperm(all_ts)
    ibackward = reverse(iforward)

    filt = make_filter(post, p)

    ysf, vysf = filter_prediction(filt, all_ts[iforward], all_ys[iforward], all_dys[iforward], all_obs[iforward])
    ysb, vysb = filter_prediction(filt, -all_ts[ibackward], all_ys[ibackward], all_dys[ibackward], all_obs[ibackward])

    ysb = reverse(ysb)
    vysb = reverse(vysb)

    ys = (vysf.*ysb + vysb.*ysf)./(vysf + vysb)
    vys = 1.0./(1.0./vysb + 1.0./vysf)

    ys_out = zeros(size(all_ts,1))
    vys_out = zeros(size(all_ts,1))

    for i in 1:size(all_ts, 1)
        ys_out[iforward[i]] = ys[i]
        vys_out[iforward[i]] = vys[i]
    end

    ys_out[size(post.ts,1)+1:end], vys_out[size(post.ts,1)+1:end]    
end

"""
    draw_extrapolation(post, p_or_x, ts, dys)

Produce an extrapolation of the process with parameters `p_or_x`
fitted to the data in `post` at the times `ts` and with reported
observational errors `dys`.

Note that the true observational uncertainty is scaled by the `nu`
parameter: `dys_true = dys*p.nu`.

`p_or_x` can be either an array or a parameter object.
"""
function draw_extrapolation(post::CARMAKalmanPosterior, x::Array{Float64, 1}, ts::Array{Float64, 1}, dys::Array{Float64, 1})
    draw_extrapolation(post, to_params(post, x), ts, dys)
end

function draw_extrapolation(post::CARMAKalmanPosterior, p::CARMAPosteriorParams, ts::Array{Float64, 1}, dys::Array{Float64, 1})
    @assert all(ts .> maximum(post.ts))
    @assert issorted(ts)

    
    filt = CARMAKalmanFilter(post, p)

    reset!(filt)

    # First, fit the filter to the observed data; by the end of the
    # loop, the filter has incorporated each data point in post.
    for i in eachindex(post.ts)
        observe!(filt, post.ys[i], p.nu*post.dys[i])

        if i < size(post.ts,1)
            advance!(filt, post.ts[i+1]-post.ts[i])
        end
    end

    # Now run the filter forward through the ts
    tcurrent = post.ts[end]
    ys_out = zeros(size(ts, 1))
    for i in eachindex(ts)
        advance!(filt, ts[i] - tcurrent)
        tcurrent = ts[i]

        draw_and_collapse!(filt)

        y, _ = predict(filt)

        ys_out[i] = y + p.nu*dys[i]*randn()
    end

    ys_out
end

type MultiSegmentCARMAKalmanPosterior
    ts::Array{Array{Float64, 1}, 1}
    ys::Array{Array{Float64, 1}, 1}
    dys::Array{Array{Float64, 1}, 1}
    p::Int
    q::Int

    function MultiSegmentCARMAKalmanPosterior(ts, ys, dys, p, q)
        @assert p>0
        @assert q<p
        @assert q>0

        new(ts, ys, dys, p, q)
    end
end

function nparams(post::MultiSegmentCARMAKalmanPosterior)
    post.p + post.q + 1 + 2*size(post.ts,1)
end

function nsegments(post::MultiSegmentCARMAKalmanPosterior)
    size(post.ts, 1)
end

type MultiSegmentCARMAPosteriorParams
    mu::Array{Float64, 1}
    sigma::Float64
    nu::Array{Float64, 1}
    arroots::Array{Complex128, 1}
    maroots::Array{Complex128, 1}
end

function to_params(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    @assert size(x, 1)==nparams(post)

    ns = nsegments(post)
    
    MultiSegmentCARMAPosteriorParams(x[1:ns], x[ns+1], x[ns+2:2*ns+1], to_roots(x[2*ns+2:2*ns+1+post.p]), to_roots(x[2*ns+2+post.p:end]))
end

function to_array(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    x = zeros(nparams(post))

    ns = nsegments(post)

    x[1:ns] = p.mu
    x[ns+1] = p.sigma
    x[ns+2:2*ns+1] = p.nu
    x[2*ns+2:2*ns+1+post.p] = to_rparams(p.arroots)
    x[2*ns+2+post.p:end] = to_rparams(p.maroots)

    x
end

function rmin_rmax(post::MultiSegmentCARMAKalmanPosterior)
    allts = sort(vcat(post.ts...))

    dt = minimum(diff(allts))
    T = allts[end] - allts[1]

    min_r = 1.0/T
    max_r = 1.0/dt

    min_r, max_r
end

function log_prior(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    ns = nsegments(post)
    if !root_params_ordered(x[2*ns+2:2*ns+1+post.p]) || !root_params_ordered(x[2*ns+2+post.p:end])
        return -Inf
    end
    log_prior(post, to_params(post, x))
end

function log_prior(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    min_r, max_r = rmin_rmax(post)

    mus = Float64[mean(y) for y in post.ys]
    sigs = Float64[std(y) for y in post.ys]

    if any(p.mu .< mus - 10.0*sigs) || any(p.mu .> mus + 10.0*sigs)
        return -Inf
    end

    sigwts = Float64[size(y,1)-1 for y in post.ys]
    sig_tot = sqrt(sum(sigwts.*sigs.*sigs)/sum(sigwts))

    if p.sigma < sig_tot / 10.0 || p.sigma > sig_tot*10.0
        return -Inf
    end

    if any(p.nu .< 0.1) || any(p.nu .> 10.0)
        return -Inf
    end

    for i in 1:post.p
        r = p.arroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    for i in 1:post.q
        r = p.maroots[i]
        if real(r) < -max_r || real(r) > -min_r
            return -Inf
        end

        if imag(r) < -max_r || imag(r) > max_r
            return -Inf
        end
    end

    -log(p.sigma) - sum(log(p.nu))
end

function make_filter(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    CARMAKalmanFilter(post, p)
end

function CARMAKalmanFilter(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    CARMAKalmanFilter(post, to_params(post, x))
end

function CARMAKalmanFilter(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    CARMAKalmanFilter(0.0, p.sigma, p.arroots, p.maroots)
end

function log_likelihood(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    log_likelihood(post, to_params(post, x))
end

function alltsysdys(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    allts = vcat(post.ts...)

    ys = Array{Float64, 1}[ys - mu for (ys, mu) in zip(post.ys, p.mu)]
    dys = Array{Float64, 1}[dys*nu for (dys, nu) in zip(post.dys, p.nu)]

    allys = vcat(ys...)
    alldys = vcat(dys...)

    inds = sortperm(allts)
    allrinds = invperm(inds)  # Gives the index in the allts array
                           # corresponding to each data point
    allts = allts[inds]
    allys = allys[inds]
    alldys = alldys[inds]

    rinds = Array{Int, 1}[]
    i = 1
    for t in post.ts
        n = size(t, 1)
        push!(rinds, allrinds[i:i+n-1])
        i = i + n
    end

    (allts, allys, alldys, rinds)
end

function log_likelihood(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    filt = nothing
    try
        filt = CARMAKalmanFilter(post, p)
    catch e
        if isa(e, DomainError)
            warn("Could not construct Kalman filter for parameters $(p)")
            return -Inf
        else
            rethrow()
        end
    end

    allts, allys, alldys, rinds = alltsysdys(post, p)

    log_likelihood(filt, allts, allys, alldys)
end

function log_posterior(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    log_posterior(post, to_params(post, x))
end

function log_posterior(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    lp = log_prior(post, p)

    if lp == -Inf
        lp
    else
        lp + log_likelihood(post, p)
    end
end

function init(post::MultiSegmentCARMAKalmanPosterior, n)
    rmin, rmax = rmin_rmax(post)

    mu0s = Float64[mean(y) for y in post.ys]
    sig0s = Float64[std(y) for y in post.ys]

    sigwts = Float64[size(y,1)-1 for y in post.ys]
    total_sig = sqrt(sum(sigwts.*sig0s.*sig0s)/sum(sigwts))

    xs = zeros((nparams(post), n))

    for i in 1:n
        mus = mu0s-10.0*sig0s + 20.0*sig0s.*rand(size(sig0s,1))
        sig = exp(log(0.1*total_sig) + rand()*(log(10.0*total_sig) - log(0.1*total_sig)))
        nus = exp(log(0.1) + rand(size(mu0s,1))*(log(10.0) - log(1.0)))

        arroots = randroots(rmin, rmax, post.p)
        maroots = randroots(rmin, rmax, post.q)

        p = MultiSegmentCARMAPosteriorParams(mus, sig, nus, arroots, maroots)
        xs[:,i] = to_array(post,p)
    end

    xs
end

function psdfreq(post::MultiSegmentCARMAKalmanPosterior; nyquist_factor=1.0, oversample_factor=1.0)
    T = post.ts[end][end] - post.ts[1][1]
    dt_med = median(diff(sort(vcat(post.ts...))))

    fmax = nyquist_factor/(2.0*dt_med)
    df = 1.0/(oversample_factor*T)

    collect(df:df:fmax)
end

function psd(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 2}, fs::Array{Float64, 1})
    psds = [psd(post, x[:,i], fs) for i in 1:size(x,2)]
    hcat(psds...)
end

function psd(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1}, fs::Array{Float64, 1})
    psd(post, to_params(post, x), fs)
end

function psd(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams, fs::Array{Float64, 1})
    psd = zeros(size(fs, 1))

    filt = CARMAKalmanFilter(post, p)

    for i in 1:size(fs, 1)
        f = fs[i]
        tpif = 2.0*pi*1.0im*f

        numer = polyeval(p.maroots, tpif) / polyeval(p.maroots, 0.0+0.0im)
        denom = polyeval(p.arroots, tpif)

        psd[i] = filt.sig*filt.sig*abs2(numer)/abs2(denom)
    end

    psd
end

function frequencies(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    frequencies(post, to_params(post, x))
end

function frequencies(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    imag(p.arroots)/(2.0*pi)
end

function frequencies(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 2})
    fs = [frequencies(post, x[:,i]) for i in 1:size(x,2)]
    hcat(fs...)
end

function drates(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    drates(post, to_params(post, x))
end

function drates(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    real(p.arroots)
end

function drates(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 2})
    drs = [drates(post, x[:,i]) for i in 1:size(x,2)]
    hcat(drs...)
end

function qfactors(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    qfactors(post, to_params(post, x))
end

function qfactors(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    fs = frequencies(post, p)
    dr = drates(post, p)

    pi*fs ./ dr
end

function qfactors(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 2})
    qs = [qfactors(post, x[:,i]) for i in 1:size(x,2)]
    hcat(qs...)
end

function residuals(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1})
    residuals(post, to_params(post, x))    
end

function residuals(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams)
    filt = CARMAKalmanFilter(post, p)

    allts, allys, alldys, rinds = alltsysdys(post, p)
    allrs, alldrs = residuals(filt, allts, allys, alldys)

    rs = [[allrs[i] for i in ri] for ri in rinds]
    drs = [[alldrs[i] for i in ri] for ri in rinds]

    (rs, drs)
end

function predict(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1}, ts::Array{Float64, 1})
    predict(post, to_params(post, x), ts)
end

function predict(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams, ts::Array{Float64, 1})
    allts, allys, alldys, rinds = alltsysdys(post, p)

    singlepost = CARMAKalmanPosterior(allts, allys, alldys, post.p, post.q)
    singleparams = to_params(singlepost, zeros(nparams(singlepost)))

    singleparams.mu = 0.0
    singleparams.sigma = p.sigma
    singleparams.nu = 1.0
    singleparams.arroots = p.arroots
    singleparams.maroots = p.maroots

    predict(singlepost, singleparams, ts)
end

function draw_extrapolation(post::MultiSegmentCARMAKalmanPosterior, x::Array{Float64, 1}, ts::Array{Float64, 1}, dys::Array{Float64, 1})
    draw_extrapolation(post, to_params(post, x), ts, dys)
end

"""
If `post` is a `MultiSegmentCARMAKalmanPosterior`, then the returned
extrapolation will have `mu = 0` and `nu = 1`.  If a different mean
and error scaling are desired they should be applied separately.
"""
function draw_extrapolation(post::MultiSegmentCARMAKalmanPosterior, p::MultiSegmentCARMAPosteriorParams, ts::Array{Float64, 1}, dys::Array{Float64, 1})
    allts, allys, alldys, rinds = alltsysdys(post, p)

    singlepost = CARMAKalmanPosterior(allts, allys, alldys, post.p, post.q)
    singleparams = to_params(singlepost, zeros(nparams(singlepost)))

    singleparams.mu = 0.0
    singleparams.sigma = p.sigma
    singleparams.nu = 1.0
    singleparams.arroots = p.arroots
    singleparams.maroots = p.maroots

    draw_extrapolation(singlepost, singleparams, ts, dys)
end


end
