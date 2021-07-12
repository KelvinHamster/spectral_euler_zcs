import numpy as np

class Integrator1D():
    """
    Provides static methods for integrators, each being
    a void function that takes a simulator and an optional atmospheric
    pressure argument, and steps it forward
    by one step. May have additional optional arguments.
    """
    @staticmethod
    def DIRK3(sim, P_atmos = None, max_iters = 100,
        tol = 10**-10):
        """
        Steps the simulation by one using Norsett's 3 stage diagonally-implicit
        Runge-Kutta method. (4th order)

        zeta is assumed to be unchanging (zeta_t = 0) over time. Implicitly
        solved by fixed point iteration, stopping after max_iters or reaching
        the desired tolerance tol (in infinity norm). If tol is not met, then
        RK4 is defaulted to. The initial guess is decided by euler's method

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array or a function of
                      (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
                      [default: constant 0]

        max_iters  -- the largest number of iterations used for fixed point
                      in each stage.

        tol        -- the desired error for each stage.
        """
        zeta_t = np.zeros(sim.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t
        
        x = 1.06858 #magic constant

        #first coeff initial guess:
        etak1,pSk1 = sim.calculate_time_derivatives(sim.eta, sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)

        tol_met = False
        for _ in range(max_iters):
            (etad,pSd) = sim.calculate_time_derivatives(
                etak1*(sim.dt*x)+sim.eta, pSk1*(sim.dt*x)+sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
            # etak1 =?= etad
            if np.linalg.norm(etak1 - etad,np.inf) \
                    + np.linalg.norm(pSk1 - pSd, np.inf) < tol:
                etak1 = etad; pSk1 = pSd
                tol_met = True
                break
            etak1 = etad; pSk1 = pSd

        if not tol_met:
            #tol not met; default to RK4
            sim.step_RK4(P_atmos)
            sim.t += sim.dt
            return

        
        #second coeff initial guess:
        etak2 = etak1; pSk2 = pSk1
        for _ in range(max_iters):
            #derivative at y_{n+1}
            (etad, pSd) = sim.calculate_time_derivatives(
                etak1*(sim.dt*(0.5-x))+etak2*(sim.dt*x)+sim.eta,
                 pSk1*(sim.dt*(0.5-x))+ pSk2*(sim.dt*x)+sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
            # etak2 =?= etad
            if np.linalg.norm(etak2 - etad,np.inf) \
                    + np.linalg.norm(pSk2 - pSd, np.inf) < tol:
                etak2 = etad; pSk2 = pSd
                tol_met = True
                break
            etak2 = etad; pSk2 = pSd

        if not tol_met:
            #tol not met; default to RK4
            sim.step_RK4(P_atmos)
            sim.t += sim.dt
            return

        
        #third coeff initial guess:
        etak3 = etak2; pSk3 = pSk2
        for _ in range(max_iters):
            #derivative at y_{n+1}
            (etad, pSd) = sim.calculate_time_derivatives(
                etak1*(sim.dt*(2*x))+etak2*(sim.dt*(1-4*x))+etak3*(sim.dt*x)
                    +sim.eta,
                 pSk1*(sim.dt*(2*x))+ pSk2*(sim.dt*(1-4*x))+ pSk3*(sim.dt*x)
                    +sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
            # etak3 =?= etad
            if np.linalg.norm(etak3 - etad,np.inf) \
                    + np.linalg.norm(pSk3 - pSd, np.inf) < tol:
                etak3 = etad; pSk3 = pSd
                tol_met = True
                break
            etak3 = etad; pSk3 = pSd

        if not tol_met:
            #tol not met; default to RK4
            sim.step_RK4(P_atmos)
            sim.t += sim.dt
            return
        
        #complete method
        y = 3*(1-2*x)**2
        sim.eta += (sim.dt*0.5/y)  * etak1
        sim.phiS += (sim.dt*0.5/y)  * pSk1
        
        sim.eta += (sim.dt*(y-1)/y)  * etak2
        sim.phiS += (sim.dt*(y-1)/y)  * pSk2

        sim.eta += (sim.dt*0.5/y)  * etak3
        sim.phiS += (sim.dt*0.5/y)  * pSk3
        sim.t += sim.dt

    @staticmethod
    def RK4(sim, P_atmos = None):
        """
        Steps the simulation by one using a 4th order Runge-Kutta method.
        zeta is assumed to be unchanging (zeta_t = 0) over time.

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array or a function of
                      (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
                      [default: constant 0]
        """
        zeta_t = np.zeros(sim.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t
        dt2 = sim.dt/2

        eta1,pS1 = sim.calculate_time_derivatives(sim.eta, sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        eta_shft = eta1*dt2; pS_shft = pS1*dt2
        eta_shft+= sim.eta; pS_shft+= sim.phiS

        eta2,pS2 = sim.calculate_time_derivatives(eta_shft, pS_shft,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        np.copyto(eta_shft,eta2); np.copyto(pS_shft,pS2)
        eta_shft *= dt2     ; pS_shft *= dt2
        eta_shft += sim.eta; pS_shft += sim.phiS

        eta3,pS3 = sim.calculate_time_derivatives(eta_shft, pS_shft,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        np.copyto(eta_shft,eta3); np.copyto(pS_shft,pS3)
        eta_shft *= sim.dt     ; pS_shft *= sim.dt
        eta_shft += sim.eta; pS_shft += sim.phiS

        eta4,pS4 = sim.calculate_time_derivatives(eta_shft, pS_shft,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)

        eta2 *= 2; pS2 *= 2; eta3 *= 2; pS3 *= 2
        eta1 += eta2; pS1 += pS2
        eta1 += eta3; pS1 += pS3
        eta1 += eta4; pS1 += pS4
        eta1 *= sim.dt/6; pS1 *= sim.dt/6
        sim.eta += eta1; sim.phiS += pS1

        sim.t += sim.dt

    @staticmethod
    def euler(sim, P_atmos = None):
        """
        Steps the simulation by one using euler's method.
        zeta is assumed to be unchanging (zeta_t = 0) over time.

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array or a function of
                      (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
                      [default: constant 0]
        """
        zeta_t = np.zeros(sim.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t

        eta1,pS1 = sim.calculate_time_derivatives(sim.eta, sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        eta1 *= sim.dt; pS1 *= sim.dt
        sim.eta += eta1; sim.phiS += pS1

        sim.t += sim.dt
    
    @staticmethod
    def implicit_midpoint(sim, P_atmos = None, max_iters = 100,
            tol = 10**-10):
        """
        Steps the simulation by one using an implicit midpoint method.
        ( y_{n+1} = y_n + h*f((y_n + y_{n+1})/2) )
        zeta is assumed to be unchanging (zeta_t = 0) over time. Implicitly
        solved by fixed point iteration, stopping after max_iters or reaching
        the desired tolerance tol (in infinity norm). If tol is not met, then
        RK4 is defaulted to. The initial guess is decided by euler's method

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array or a function of
                      (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
                      [default: constant 0]

        max_iters  -- the largest number of iterations used for fixed point.

        tol        -- the desired error.
        """
        zeta_t = np.zeros(sim.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t

        #initial guess
        eta1,pS1 = sim.calculate_time_derivatives(sim.eta, sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        eta1 *= sim.dt; pS1 *= sim.dt
        eta1 += sim.eta; pS1 += sim.phiS
        
        for i in range(max_iters):
            #derivative of average
            (delta_e, delta_p) = sim.calculate_time_derivatives(
                (sim.eta + eta1)/2,
                (sim.phiS + pS1)/2,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
            #next guess, calculate error
            delta_e *= sim.dt; delta_p *= sim.dt
            delta_e += sim.eta; delta_p += sim.phiS
            if np.linalg.norm(eta1 - delta_e,np.inf) \
                    + np.linalg.norm(pS1 - delta_p, np.inf) < tol:
                sim.eta = delta_e; sim.phiS = delta_p
                sim.t += sim.dt
                return

            eta1 = delta_e; pS1 = delta_p

        #tol not met; default to RK4
        sim.step_RK4(P_atmos)
        sim.t += sim.dt

    @staticmethod
    def AM1(sim, P_atmos = None, max_iters = 100,
            tol = 10**-10):
        """
        Steps the simulation by one using an implicit trapezoidal rule.
        (Adams-Moulton 1)
        ( y_{n+1} = y_n + h/2*(f(y_n) + f(y_{n+1})) )
        zeta is assumed to be unchanging (zeta_t = 0) over time. Implicitly
        solved by fixed point iteration, stopping after max_iters or reaching
        the desired tolerance tol (in infinity norm). If tol is not met, then
        RK4 is defaulted to. The initial guess is decided by euler's method

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array or a function of
                      (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
                      [default: constant 0]

        max_iters  -- the largest number of iterations used for fixed point.

        tol        -- the desired error.
        """
        zeta_t = np.zeros(sim.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t

        #initial guess
        etad1,pSd1 = sim.calculate_time_derivatives(sim.eta, sim.phiS,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
        eta1 = etad1*sim.dt; pS1 = etad1*sim.dt
        eta1 += sim.eta; pS1 += sim.phiS
        for i in range(max_iters):
            #derivative at y_{n+1}
            (etad2, pSd2) = sim.calculate_time_derivatives(
                eta1, pS1,
                sim.zeta, sim.zeta_x, zeta_t, P_atmos)
            # eta1 =?= eta + h/2 (etad1 + etad2)

            #next guess, calculate error
            etad2 += etad1; pSd2 += pSd1
            etad2 *= sim.dt/2; pSd2 *= sim.dt/2
            etad2 += sim.eta; pSd2 += sim.phiS
            if np.linalg.norm(eta1 - etad2,np.inf) \
                    + np.linalg.norm(pS1 - pSd2, np.inf) < tol:
                sim.eta = etad2; sim.phiS = pSd2
                sim.t += sim.dt
                return

            eta1 = etad2; pS1 = pSd2

        #tol not met; default to RK4
        sim.step_RK4(P_atmos)
        sim.t += sim.dt
