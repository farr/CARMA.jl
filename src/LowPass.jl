""" Implements an exponentially-weighted smoothing method for low-pass
filtering.

For a data set `ys` at times `ts` with observational uncertainties `dys`, the
output point ``\\tilde{y}_i`` is given by the weighted sum

``\\tilde{y}_i = \\frac{\\sum_{\\left. j \\mid t_j < t_i \\right.} w_{j,i} y_j + y_i + \\sum_{\\left. k \\mid t_k > t_i \\right.} w_{k,i} y_k}{1 + \\sum_{\\left. j \\mid t_j < t_i \\right.} w_{j,i} + \\sum_{\\left. k \\mid t_k > t_i \\right.} w_{k,i}}``

and the uncertainty on the output point is given by

``\\sigma_{\\tilde{y}_i} = \\sqrt{\\frac{\\sum_{\\left. j \\mid t_j < t_i \\right.} w_{j,i}^2 \\sigma_{y_j}^2 + \\sigma_{y_i}^2 + \\sum_{\\left. k \\mid t_k > t_i \\right.} w_{k,i}^2 \\sigma_{y_k}^2}{\\left(1 + \\sum_{\\left. j \\mid t_j < t_i \\right.} w_{j,i} + \\sum_{\\left. k \\mid t_k > t_i \\right.} w_{k,i}\\right)^2}}``

where the weights are given by the exponential

``w_{j,i} = \\exp\\left( -\\frac{\\left| t_j - t_i \\right|}{\\tau} \\right)``

and similarly for ``w_{k,i}``.  Because

``w_{j,i+1} = w_{j,i} \\exp\\left( -\\frac{\\left| t_{i+1} - t_i \\right|}{\\tau}\\right)``

for all ``j < i``, we have

``\\sum_{\\left. j \\mid t_j < t_{i+1} \\right.} w_{j,i+1} y_j = \\exp\\left( -\\frac{\\left| t_{i+1} - t_i \\right|}{\\tau}\\right) \\left[ \\sum_{\\left. j \\mid t_j < t_i \\right.} w_{j,i} y_j + y_i \\right]``

and similarly for all other sums appearing above.  These recursions enable the
computation of all the terms appearing in the outputs above during one forward
and one reverse pass over the data.

The filter has an (exact) autocorrelation time of ``2\\tau``, which permits
thinning of the output time series to obtain (approximately) independent
uncertainties on the low-passed data.

"""
module LowPass

"""
    exp_lowpass(ts, ys, dys, tau)

Low-pass filter the input data `ys` taken at times `ts` with uncertainties `dys`
with exponential weighting with timescale `tau`.

Returns `(ys_lowpass, dys_lowpass)`, sampled at the same rate as the original
series.  (But see `thin_exp_lowpass`.)
"""
function exp_lowpass(ts, ys, dys, tau)
    n = size(ts, 1)

    left_mus = zeros(n)
    left_sigmas = zeros(n)
    left_sums = zeros(n)

    right_mus = zeros(n)
    right_sigmas = zeros(n)
    right_sums = zeros(n)

    for i in 2:n
        dt = ts[i] - ts[i-1]
        wt = exp(-dt/tau)

        left_mus[i] = left_mus[i-1]*wt + wt*ys[i-1]
        left_sigmas[i] = wt*sqrt(left_sigmas[i-1]*left_sigmas[i-1] + dys[i-1]*dys[i-1])
        left_sums[i] = wt*left_sums[i-1] + wt
    end

    for i in n-1:-1:1
        dt = ts[i+1]-ts[i]
        wt = exp(-dt/tau)

        right_mus[i] = right_mus[i+1]*wt + wt*ys[i+1]
        right_sigmas[i] = wt*sqrt(right_sigmas[i+1]*right_sigmas[i+1] + dys[i+1]*dys[i+1])
        right_sums[i] = wt*right_sums[i+1] + wt
    end

    ys_filt = (left_mus .+ right_mus .+ ys)./(1.0 .+ left_sums .+ right_sums)
    dys_filt = sqrt.((left_sigmas.*left_sigmas .+ right_sigmas.*right_sigmas .+ dys.*dys)./((1.0 .+ left_sums .+ right_sums).*(1.0 .+ left_sums .+ right_sums)))

    ys_filt, dys_filt
end

"""
    thin_exp_lowpass(ts, ys, dys, tau)

Return `(ts_thin, ys_thin, dys_thin)` with data chosen from the input times
`ts`, samples `ys`, and uncertainties `dys` but spaced by at least `tau` in
time.

The combination of `exp_lowpass` and `thin_exp_lowpass` will give
(approximately) independent-uncertainty low-passed samples from a time series.
"""
function thin_exp_lowpass(ts, ys, dys, tau)
    ifill = 1
    ts_t = [ts[1]]
    ys_t = [ys[1]]
    dys_t = [dys[1]]

    for i in 2:size(ts, 1)
        if ts[i] > ts_t[ifill] + 2*tau
            push!(ts_t, ts[i])
            push!(ys_t, ys[i])
            push!(dys_t, dys[i])
            ifill += 1
        end
    end

    ts_t, ys_t, dys_t
end

end
