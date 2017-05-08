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

type CeleriteKalmanFilter
    mu::Float64 # Mean
    x::Array{Complex128, 1} # State mean
    Vx::Array{Complex128, 2} # State variance
    K::Array{Complex128, 1} # Kalman Gain
    lambda::Array{Complex128, 1} # Evolution factors exp(roots*dt)
    b::Array{Complex128, 2} # Rotated observation vector
    roots::Array{Complex128, 1} # Eigenvalues of the ODEs
    V::Array{Complex128, 2} # Stationary covariance
    Vtemp::Array{Complex128, 2} # Storage for matrix ops
end

function reset!(filt::CeleriteKalmanFilter)
    p = size(filt.x, 1)
    filt.x = zeros(Complex128, p)
    filt.Vx = copy(filt.V)
    filt.K = zeros(Complex128, p)
    filt
end

function osc_roots(osc_freqs::Array{Float64, 1}, osc_Qs::Array{Float64, 1})
    nosc = size(osc_freqs, 1)

    oscroots = zeros(Complex128, nosc)
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

    roots = zeros(Complex128, dim)
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

    b = zeros(Complex128, dim)
    
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

    U = zeros(Complex128, (dim,dim))

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

    e = zeros(Complex128, dim)
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

    V = zeros(Complex128, (dim,dim))

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

    CeleriteKalmanFilter(mu, zeros(Complex128, dim), V, zeros(Complex128, dim), zeros(Complex128, dim), b, roots, copy(V), zeros(Complex128, (dim,dim)))
end

function advance!(filt::CeleriteKalmanFilter, dt::Float64)
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
            filt.Vx[i,j] = lam[i]*conj(lam[j])*(filt.Vx[i,j] - filt.V[i,j]) + filt.V[i,j]
        end
    end

    filt
end

function observe!(filt::CeleriteKalmanFilter, y::Float64, dy::Float64)
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
            filt.Vx[i,j] = filt.Vx[i,j] - vy*filt.K[i]*conj(filt.K[j])
        end
    end

    filt
end

function predict(filt::CeleriteKalmanFilter)
    p = size(filt.x,1)
    
    yp = filt.mu
    for i in 1:p
        yp += real(filt.b[1,i]*filt.x[i])
    end

    vyp = 0.0
    for i in 1:p
        for j in 1:p
            vyp += real(filt.b[1,i]*filt.Vx[i,j]*conj(filt.b[1,j]))
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
        filt.Vx = zeros(Complex128, (nd, nd))
    catch e
        if isa(e, Base.LinAlg.PosDefException)
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
            filt.Vx = zeros(Complex128, (nd, nd))
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
        cov += drw_rms[i]*drw_rms[i]*exp(-drw_rates[i]*dts)
    end

    for i in 1:nosc
        A = 1.0 / (-4.0*real(oscroots[i])*(conj(oscroots[i]) - oscroots[i])*oscroots[i])
        B = 1.0 / (-4.0*real(oscroots[i])*(oscroots[i] - conj(oscroots[i]))*conj(oscroots[i]))

        s2 = osc_rms[i]*osc_rms[i] / (A+B)
        cov += real(s2*(A*exp(oscroots[i]*dts) + B*exp(conj(oscroots[i])*dts)))
    end

    for i in 1:N
        cov[i,i] += dys[i]*dys[i]
    end

    cov
end
end
