# This is a simple test file that checks (on generated data) whether the
# Kalman.jl functions for AR(1), CARMA(p,q), and multi-epoch CARMA(p,q) can
# accurately reconstruct the parameters from a pre-generated data set.

using DelimitedFiles
using Ensemble
using HDF5
using Statistics

@testset "Kalman.jl Test Suite" begin
    @testset "AR(1) Fit Test" begin
        ar1_parameters = Dict("sigma" => Float64(pi), "mu" => 123.32, "tau" => sqrt(2), "nu" => 0.77)
        data = readdlm("ar1_data.txt")
        ts = data[:,1]
        ys = data[:,2]
        dys = data[:,3]

        post = Kalman.AR1KalmanPosterior(ts, ys, dys)

        pts = Kalman.init(post, 128)
        ns = EnsembleNest.NestState(x->Kalman.log_likelihood(post,x), x->Kalman.log_prior(post,x), pts, 128)
        EnsembleNest.run!(ns, 0.1, verbose=true)
        ps, lls = EnsembleNest.postsample(ns)

        mus = [Kalman.to_params(post, ps[:,j]).mu for j in 1:size(ps,2)]
        sigmas = [Kalman.to_params(post, ps[:,j]).sigma for j in 1:size(ps,2)]
        taus = [Kalman.to_params(post, ps[:,j]).tau for j in 1:size(ps,2)]
        nus = [Kalman.to_params(post, ps[:,j]).nu for j in 1:size(ps,2)]

        @test abs(mean(mus) - ar1_parameters["mu"]) < 2*std(mus)
        @test abs(mean(sigmas) - ar1_parameters["sigma"]) < 2*std(sigmas)
        @test abs(mean(taus) - ar1_parameters["tau"]) < 2*std(taus)
        @test abs(mean(nus) - ar1_parameters["nu"]) < 2*std(nus)
    end

    @testset "CARMA Fit Test" begin
        mu = 123.32
        sigma = Float64(pi)
        nu = 0.77
        f0 = 0.866
        r0 = 1.0/1.33
        b1 = 1.0/0.235
        b2 = 1.0/1.57
        Q = 10.0
        arroots = [-r0, -f0/Q+f0*1im, -f0/Q-f0*1im]
        maroots = [-b1, -b2]

        data = readdlm("carma_data.txt")
        ts = data[:,1]
        ys = data[:,2]
        dys = data[:,3]

        post = Kalman.CARMAKalmanPosterior(ts, ys, dys, 3, 2)
        pts = Kalman.init(post, 128)
        ns = EnsembleNest.NestState(x->Kalman.log_likelihood(post,x), x->Kalman.log_prior(post,x), pts, 128)
        EnsembleNest.run!(ns, 0.1, verbose=true)
        ps, lls = EnsembleNest.postsample(ns)

        mus = [Kalman.to_params(post, ps[:,j]).mu for j in 1:size(ps,2)]
        sigmas = [Kalman.to_params(post, ps[:,j]).sigma for j in 1:size(ps,2)]
        nus = [Kalman.to_params(post, ps[:,j]).nu for j in 1:size(ps,2)]
        fs = vec(maximum(Kalman.frequencies(post, ps), dims=1))

        @test abs(mean(mus) - mu) < 2*std(mus)
        @test abs(mean(sigmas) - sigma) < 2*std(sigmas)
        @test abs(mean(nus) - nu) < 2*std(nus)
        @test abs(mean(fs) - f0) < 2*std(fs)
    end

    @testset "Multi-Segment CARMA Fit" begin
        mu = 123.32
        sigma = Float64(pi)
        nu = 0.77
        f0 = 0.866
        r0 = 1.0/1.33
        b1 = 1.0/0.235
        b2 = 1.0/1.57
        Q = 10.0
        arroots = [-r0, -f0/Q+f0*1im, -f0/Q-f0*1im]
        maroots = [-b1, -b2]

        data = readdlm("carma_data.txt")
        ts = data[:,1]
        ys = data[:,2]
        dys = data[:,3]

        dmu2 = -25.3234
        dmu3 = +136.0

        nufac2 = 1.33
        nufac3 = 0.756

        ts = [ts[1:3:end], ts[2:3:end], ts[3:3:end]]
        ys = [ys[1:3:end], ys[2:3:end].+dmu2, ys[3:3:end].+dmu3]
        dys = [dys[1:3:end], dys[2:3:end].*nufac2, dys[3:3:end].*nufac3]

        post = Kalman.MultiSegmentCARMAKalmanPosterior(ts, ys, dys, 3, 2)
        pts = Kalman.init(post, 128)
        ns = EnsembleNest.NestState(x->Kalman.log_likelihood(post,x), x->Kalman.log_prior(post,x), pts, 128)
        EnsembleNest.run!(ns, 0.1, verbose=true)
        ps, lls = EnsembleNest.postsample(ns)

        mus = hcat([Kalman.to_params(post, ps[:,j]).mu for j in 1:size(ps,2)]...)
        nus = hcat([Kalman.to_params(post, ps[:,j]).nu for j in 1:size(ps,2)]...)
        sigmas = [Kalman.to_params(post, ps[:,j]).sigma for j in 1:size(ps, 2)]
        fs = vec(maximum(Kalman.frequencies(post, ps), dims=1))

        mu0 = [mu, mu+dmu2, mu+dmu3]
        nu0 = [nu, nu/nufac2, nu/nufac3]

        for i in 1:3
            @test abs(mean(mus[i,:]) - mu0[i]) < 2*std(mus[i,:])
            @test abs(mean(nus[i,:]) - nu0[i]) < 2*std(nus[i,:])
        end
        @test abs(mean(sigmas) - sigma) < 2*std(sigmas)
        @test abs(mean(fs) - f0) < 2*std(fs)
    end

    @testset "Read/write to HDF5" begin
        N = 128
        p = 3
        q = 2
        ts = sort(randn(N))
        ys = randn(N)
        dys = abs.(randn(N))

        post = Kalman.CARMAKalmanPosterior(ts, ys, dys, p, q)

        try
            h5open("test-CARMA-Kepler.h5", "w") do f
                write(f, post)
            end

            h5open("test-CARMA-Kepler.h5", "r") do f
                global post2 = Kalman.CARMAKalmanPosterior(f)
            end

            @test all(post.ts .== post2.ts)
            @test all(post.ys .== post2.ys)
            @test all(post.dys .== post2.dys)
            @test post.p == post2.p
            @test post.q == post2.q
        finally
            rm("test-CARMA-Kepler.h5")
        end
    end
end
