@testset "Low Pass Tests" begin
    @testset "Agreement with explicit smoothing for random data" begin
        ts = cumsum(rand(1000))
        ys = randn(1000)
        dys = 0.1 + rand(1000)

        i = 100
        tau = pi

        wts = exp.(-abs.(ts-ts[i])/tau)
        yout = sum(ys.*wts)/sum(wts)
        dyout = sqrt(sum(wts.*wts.*dys.*dys)/(sum(wts)*sum(wts)))

        ys_filt, dys_filt = LowPass.exp_lowpass(ts, ys, dys, tau)

        @test abs(ys_filt[i] - yout) < 1e-10
        @test abs(dys_filt[i] - dyout) < 1e-10
    end
end
