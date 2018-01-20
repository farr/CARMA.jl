module Kepler

function ecc_anom(M, e)
    M = mod2pi(M)
    if M < zero(M)
        M = M + 2*pi
    end

    Emin = zero(M)
    fmin = nothing
    Emax = 2*pi*one(M)
    fmax = nothing
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

function true_anom(ea, e)
    ta = 2*atan(sqrt((one(e)+e)/(one(e)-e))*tan(ea/2))
    if ta < zero(ta)
        ta + 2*pi
    else
        ta
    end
end

function rv(t, K, P, e, omega, chi)
    fp, ip = modf(t/P - chi)

    if fp < zero(fp)
        fp = fp + one(fp)
    end

    m = 2*pi*fp
    
    ta = true_anom(ecc_anom(m, e), e)

    K*(cos(ta + omega) + e*cos(omega))
end

end
