module Kepler

function ecc_anom(M::Float64, e::Float64)
    M = mod2pi(M)
    if M < zero(M)
        M = M + 2.0*pi
    end

    Emin = zero(M)
    fmin = zero(M)
    Emax = 2.0*pi*one(M)
    fmax = zero(M)
    Eguess = M
    while true
        f = M - Eguess + e*sin(Eguess)

        if abs(f) < convert(typeof(f), 1e-10)
            return Eguess
        end
        df = -one(Eguess) + e*cos(Eguess)

        Enewton = Eguess - f/df

        if Enewton < Emin || Enewton > Emax
            fmin = M - Emin - e*sin(Emin)
            fmax = M - Emax - e*sin(Emax)

            if fmin == zero(fmin)
                return Emin
            elseif fmax == zero(fmin)
                return Emax
            end

            if fmin*f < zero(fmin)
                Emax = Eguess
            elseif fmax*f < zero(fmax)
                Emin = Eguess
            else
                error("Not bracketing a root in bisection.")
            end
            Eguess = (Emax-Emin)/2
        else
            Eguess = Enewton
        end
    end
end

function true_anom(ea::Float64, e::Float64)
    ta = 2.0*atan(sqrt((one(e)+e)/(one(e)-e))*tan(ea/2.0))
    if ta < zero(ta)
        ta + 2.0*pi
    else
        ta
    end
end

function rv(t::Float64, K::Float64, P::Float64, e::Float64, omega::Float64, chi::Float64)
    fp, ip = modf(t/P - chi)

    if fp < zero(fp)
        fp = fp + one(fp)
    end

    m = 2*pi*fp

    ta = true_anom(ecc_anom(m, e), e)

    K*(cos(ta + omega) + e*cos(omega))
end

end
