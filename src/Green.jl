module Green

export alphas, full_cov

""" Returns the coefficients giving the linear combination of values
satisfying the homogeneous equation with the given roots.  For `p`
roots, the first `p` alphas are zero, and the remaining will satisfy

    y[k] - sum([alphas[i,k]*y[k-i] for i in 1:p]) == 0

when `y` satisfies the homogeneous equation.  """
function alphas(ts::Array{Float64, 1}, roots::Array{ComplexF64, 1})
    p = size(roots, 1)
    n = size(ts, 1)

    alphas = zeros(p, n)

    M = zeros(ComplexF64, p,p)
    b = ones(p)

    # Alphas for samples before the p+1-st are zero
    for j in p+1:n
        for k in 1:p
            for l in 1:p
                M[k,l] = exp(roots[k]*(ts[j-(p-l+1)]-ts[j]))
            end
        end
        alphas[:,j] = real(M\b)
    end

    alphas
end

""" Returns the coefficients `g' of the exponential solutions in the
Green's function:

    g(t,xi) == sum([g[i]*exp(roots[i]*(t-xi)) for i in 1:p]) """
function greens_coeff(roots::Array{ComplexF64, 1})
    p = size(roots, 1)

    M = zeros(ComplexF64, p, p)
    b = zeros(p)

    for i in 1:p
        for j in 1:p
            M[i,j] = roots[j]^(i-1)
        end
    end

    b[end] = 1.0

    M\b
end

""" Returns the coefficients of the exponential terms in the Green's
function integrand for the solution `y`:

    c[i] == g[i]*prod([(roots[i] + maroots[j])/(-maroots[j]) for j in 1:q])

where `y` satisfies the ODE

    prod([(ddt - roots[i]) for i in 1:p])*y(t) == prod([(ddt - maroots[i])/(-maroots[i]) for i in 1:q])*eta(t)

with `eta` a white noise process with variance `sigma^2`.

"""
function greens_integrand_coeff(roots::Array{ComplexF64,1}, maroots::Array{ComplexF64, 1}, g::Array{ComplexF64, 1})
    p = size(roots, 1)
    q = size(maroots, 1)

    if q == 0
        g
    else
        coeff = zeros(ComplexF64, p)

        for i in 1:p
            coeff[i] = g[i]
            for j in 1:q
                coeff[i] *= (roots[i] + maroots[j])/(-maroots[j])
            end
        end

        coeff
    end
end

function cov_coeffs(gic::Array{ComplexF64, 1}, roots::Array{ComplexF64, 1})
    p = size(roots, 1)

    cc = zeros(ComplexF64, p)

    for i in 1:p
        cc[i] = zero(cc[i])
        for j in 1:p
            cc[i] -= gic[i]*conj(gic[j])/(roots[i] + conj(roots[j]))
        end
    end

    cc
end

""" Returns the full covariance matrix computed using the Green's
function method at the given times, `ts`.  Requires that the `ts`
array is sorted to make bookkeeping easier.  """
function full_cov(ts::Array{Float64, 1}, sigma::Float64, roots::Array{ComplexF64, 1}, maroots::Array{ComplexF64, 1})
    n = size(ts, 1)
    p = size(roots, 1)
    q = size(maroots, 1)

    for i in 1:n-1
        @assert ts[i] < ts[i+1]
    end

    g = greens_coeff(roots)
    gi = greens_integrand_coeff(roots, maroots, g)

    ccs = cov_coeffs(gi, roots)

    cov = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:i
            celt = zero(ComplexF64)
            dt = ts[i] - ts[j]
            for k in 1:p
                celt += ccs[k]*exp(roots[k]*dt)
            end
            cov[i,j] = real(celt)
            cov[j,i] = cov[i,j]
        end
    end

    sigma2 = sigma*sigma
    factor = sigma2 / cov[1,1]

    for j in 1:n
        for i in 1:n
            cov[i,j] = cov[i,j] * factor
        end
    end

    cov
end

end
