@testset "Celerite Tests" begin
    @testset "Celerite Kalman and Raw Log-Likelihoods Agree" begin
        # Test that the raw and Kalman log-likelihood agree
        # The data are a snippet from KIC 011615890
        data = readdlm("test-data.dat")
        test_ts = data[:,1]
        test_fs = data[:,2]
        test_dfs = data[:,3]

        drw_rms = [sqrt(1000.0)]
        drw_rates = [1.0]
        osc_rms = [sqrt(1432.0)]
        osc_freqs = [18.0]
        osc_Qs = [100.0]

        mu = 210933.0

        N = size(test_ts,1)

        filt = Celerite.CeleriteKalmanFilter(mu, drw_rms, drw_rates, osc_rms, osc_freqs, osc_Qs)
        ll_filt = Celerite.log_likelihood(filt, test_ts, test_fs, test_dfs)
        cov = Celerite.raw_covariance(test_ts, test_dfs, drw_rms, drw_rates, osc_rms, osc_freqs, osc_Qs)
        F = cholfact(cov)
        L = F[:L]
        logdet = 0.0
        for i in 1:size(cov,1)
            logdet += log(L[i,i])
        end
        ll_raw = -0.5*N*log(2.0*pi) - logdet - 0.5*dot((test_fs - mu), F \ (test_fs - mu))

        @test abs(ll_filt - ll_raw[1]) < 1e-10
    end
end
