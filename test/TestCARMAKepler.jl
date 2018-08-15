using Base.Filesystem
using DelimitedFiles
using Ensemble
using Statistics

@testset "CARMAKepler.jl Tests" begin
    @testset "Aldebaran Test Run" begin
        ts = Array{Float64, 1}[]
        ys = Array{Float64, 1}[]
        dys = Array{Float64, 1}[]

        for i in 3:9
            d = readdlm("aldeb-data/$(i)/table$(i).dat")
            push!(ts, d[:,1])
            push!(ys, d[:,2])
            push!(dys, d[:,3])
        end
        d = readdlm("aldeb-data/song/tablesong.dat")
        push!(ts, d[:,1])
        push!(ys, d[:,2])
        push!(dys, d[:,3])

        # Re-center ts
        tmed = median(vcat(ts...))
        ts = [t.-tmed for t in ts]

        tmin = minimum([minimum(t) for t in ts])
        tmax = maximum([maximum(t) for t in ts])
        T = tmax - tmin
        dt_min = minimum([minimum(diff(t)) for t in ts])
        muHz = 1e-6*(3600.0*24.0)

        post = CARMAKepler.MultiEpochPosterior(ts, ys, dys, 600.0, 700.0, 1, 1, 1.0/(2*T), 0.05*muHz, 0.1*muHz, 10*muHz, 1000.0);
        pts = CARMAKepler.draw_prior(post, 1024)

        ns = EnsembleNest.NestState(x->CARMAKepler.log_likelihood(post, x), x->CARMAKepler.log_prior(post, x), pts, 128)

        EnsembleNest.run!(ns, 0.1, verbose=true)

        ps, lls = EnsembleNest.postsample(ns)

        Ps = [CARMAKepler.to_params(post, ps[:,j]).P for j in 1:size(ps,2)]
        Ks = [CARMAKepler.to_params(post, ps[:,j]).K for j in 1:size(ps,2)]
        fs = [CARMAKepler.to_params(post, ps[:,j]).osc_freq[1] for j in 1:size(ps,2)]

        @test abs(mean(Ps) - 628) < 10
        @test abs(mean(Ks) - 125) < 25
        @test abs(mean(fs)/muHz - 2.2) < 0.2
    end
end
