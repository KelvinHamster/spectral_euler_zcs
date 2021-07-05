import math
import numpy as np

class Simulator1D():
    """
    Given the necessary states to step the simulation, simulates the wave,
    using only properties of the surface at each step. Transcribed from
    Matlab.
    
    original from Knowles & Yeh (2018)
    """

    def __init__(self, bathymetry, dt, dx, eta0, phiS0, zeta_x = None,
            M = 5, v = 0.7, g = 9.81, h0 = None):
        """
        Initializes a Simulator instance with the given parameters:
            bathymetry  --  numpy array of the bathymetry, must have an even
                            number of nodes. 0 is expected to be water level.
            dt          --  time resolution of the simulation
            dx          --  spatial resolution of the simulation (distance
                            between points of bathymetry)
            eta0        --  initial free surface heights (with spatial
                            resolution dx, 0 is expected for still water)
                            expected to be a numpy array
            phiS0       --  velocity potential at the free surface, expected
                            to be a numpy array.

        Has the following keyword arguments:

            M    -- Terms in the pertubation expansion (higher number is more
                    accurate, but requires more computation) [default: 5]
            v    -- lowpass threshold. Any wavenumber greater than the largest
                    wavenumber times v is clipped off each timestep. Expects
                    a float in (0,1); [default: 0.7]
            g    -- acceleration due to gravity. [default: 9.81]
            h0   -- base still water depth [default: bathymetry[0]]
        """
        self.dt = dt; self.dx = dx
        self.eta = eta0; self.phiS = phiS0
        
        self.M = M; self.v = v; self.g = g
        if h0 == None:
            self.h0 = -bathymetry[0]
        else:
            self.h0 = h0
        
        self.zeta = bathymetry + self.h0
        
        # length
        self.Nx = len(bathymetry)
        self.sim_length = self.Nx * dx # total length

        # each x corresponds with a half open interval [--) of
        # size dx
        self.x = np.linspace(0,(self.sim_length-dx),self.Nx)

        # initialize wavenumber: use double-domains, should match
        # domain of FFT
        self.kxdb = np.array(
            list(range(self.Nx)) + list(range(-self.Nx, 0)),
            np.float64
        )
        self.kxdb *= math.pi/self.sim_length
        #magnitude of wavenumber
        self.kappadb = np.abs(self.kxdb)
        #lowpass filter; multiply to use
        self.chi = self.kappadb <= (v*max(self.kappadb))
        #ik for fast reference when differentiating
        self.kxdb_im = self.kxdb * complex(0,1)
        self.kxdb_im *= self.chi
        
        if zeta_x == None:
            # gradient of bathymetry; just use divided difference
            self.zeta_x = np.zeros(self.Nx)
            self.zeta_x[1:-1] = (self.zeta[2:] - self.zeta[:-2])/(2*dx)

            # fourier_zetadb = np.fft.fft(np.append(self.zeta, 
            #           np.fliplr([self.zeta])))
            # fourier_zetadb *= self.kxdb_im
            # self.zeta_x = np.real(np.fft.ifft(fourier_zetadb))[0:self.Nx]
        else:
            self.zeta_x = zeta_x

        self.t = 0 # time in the simulation

    def step_RK4(self, P_atmos = None):
        """
        Steps the simulation by one using a 4th order Runge-Kutta method.
        zeta and atmospheric pressure is assumed to be unchanging (zeta_t = 0)
        over time.

        P_atmos    -- atmospheric pressure at every point, should have the
                      same samples as bathymetry. Expected to be a numpy
                      array [default: constant 0]
        """
        zeta_t = np.zeros(self.Nx)
        if P_atmos == None:
            #zeta_t does not change, so we can just go by reference
            P_atmos = zeta_t
        dt2 = self.dt/2

        eta1,pS1 = self.calculate_time_derivatives(self.eta, self.phiS,
                self.zeta, self.zeta_x, zeta_t, P_atmos)
        eta_shft = eta1*dt2; pS_shft = pS1*dt2
        eta_shft+= self.eta; pS_shft+= self.phiS

        eta2,pS2 = self.calculate_time_derivatives(eta_shft, pS_shft,
                self.zeta, self.zeta_x, zeta_t, P_atmos)
        np.copyto(eta_shft,eta2); np.copyto(pS_shft,pS2)
        eta_shft *= dt2     ; pS_shft *= dt2
        eta_shft += self.eta; pS_shft += self.phiS

        eta3,pS3 = self.calculate_time_derivatives(eta_shft, pS_shft,
                self.zeta, self.zeta_x, zeta_t, P_atmos)
        np.copyto(eta_shft,eta3); np.copyto(pS_shft,pS3)
        eta_shft *= self.dt     ; pS_shft *= self.dt
        eta_shft += self.eta; pS_shft += self.phiS

        eta4,pS4 = self.calculate_time_derivatives(eta_shft, pS_shft,
                self.zeta, self.zeta_x, zeta_t, P_atmos)

        eta2 *= 2; pS2 *= 2; eta3 *= 2; pS3 *= 2
        eta1 += eta2; pS1 += pS2
        eta1 += eta3; pS1 += pS3
        eta1 += eta4; pS1 += pS4
        eta1 *= self.dt/6; pS1 *= self.dt/6
        self.eta += eta1; self.phiS += pS1

        self.t += dt
        


    def calculate_time_derivatives(self, eta, phiS, zeta, zeta_x, zeta_t, P_a):
        """
        returns (eta_t, phiS_t), a tuple of the time derivatives of eta and
        phiS. Takes in the following values:

        eta      -- free surface height at the given time step; pass
                    simulator.eta if you want the current time derivative.
        phiS     -- free surface velocity potential at the given time step;
                    pass simulator.phiS if you want the current time derivative
        zeta     -- bathymetry, with 0 corresponding with a depth of -h0
        zeta_x   -- spatial derivative of zeta
        zeta_t   -- time derivative of zeta
        P_a      -- atmospheric pressure at a given x.
        """
        #gradients:
        fourier_edb = np.fft.fft(np.append(eta, np.fliplr([eta])[0]))
        fourier_edb *= self.kxdb_im
        eta_x = np.real(np.fft.ifft(fourier_edb))[0:self.Nx]

        fourier_pSdb = np.fft.fft(np.append(phiS, np.fliplr([phiS])[0]))
        fourier_pSdb *= self.kxdb_im
        phiS_x = np.real(np.fft.ifft(fourier_pSdb))[0:self.Nx]

        w = self.vertvel(eta, phiS, zeta, zeta_x, zeta_t)

        eta_x_sq_p1 = eta_x**2
        eta_x_sq_p1 += 1
        # not sure if these are in-place operations;
        # if not, this can be optimized.
        return (
            -phiS_x*eta_x + eta_x_sq_p1*w,
            -P_a - self.g*eta - (phiS_x**2)/2 + eta_x_sq_p1/2*(w**2)
        )


    def vertvel(self, eta, phiS, zeta, zeta_x, zeta_t):
        """
        Calculate phi_z at the free surface
        """
        dadzfs_mat = np.zeros((self.M, self.M, self.Nx*2))
        dadzdxbb_mat = np.zeros((self.M, self.M, self.Nx*2))
        dadzbb_mat = np.zeros((self.M, self.M, self.Nx*2))
        
        dbdzfs_mat = np.zeros((self.M, self.M, self.Nx*2))
        dbdzdxbb_mat = np.zeros((self.M, self.M, self.Nx*2))
        dbdzbb_mat = np.zeros((self.M, self.M, self.Nx*2))

        A_m_z0 = np.copy(phiS) #A_1 evaluated at z=0
        A_m_coeff = np.fft.fft(np.append(A_m_z0, np.fliplr([A_m_z0])[0]))
        A_m_coeff *= self.chi #coefficient array

        Bz_m_zh = np.copy(zeta_t) #(B_1)_z at z=-h, only used to get the coeffs
        B_m_coeff = np.fft.fft(np.append(Bz_m_zh, np.fliplr([Bz_m_zh])[0]))
        B_m_coeff *= self.chi #coefficient array

        
        j_indices = np.array(range(self.M))

        #we will use kh alot
        kxdbh = self.kxdb * self.h0
        tanh_kh = np.tanh(kxdbh)
        sech_kh = np.cosh(kxdbh); sech_kh **= -1

        for m in range(self.M):

            for j in range(self.M - m):
                kxdbj = self.kxdb ** j
                kxdbjp1 = kxdbj * self.kxdb

                if j % 2 == 1:
                    dadzfs_mat[m][j] = np.real(np.fft.ifft(
                        A_m_coeff *
                        (kxdbjp1)
                    ))
                    
                    dbdzdxbb_mat[m][j] = \
                    np.real(np.fft.ifft(
                        complex(0,1) * B_m_coeff *
                        (kxdbj)
                    ))

                    dbdzbb_mat[m][j] = \
                    np.real(np.fft.ifft(
                        B_m_coeff *
                        (kxdbjp1)
                    ))
                else:
                    dadzfs_mat[m][j] = np.real(np.fft.ifft(
                        A_m_coeff *
                        (kxdbjp1)
                        *(tanh_kh)
                    ))

                    dadzdxbb_mat[m][j] = \
                    np.real(np.fft.ifft(
                        A_m_coeff * self.kxdb_im *
                        (kxdbj)
                        *sech_kh
                    ))

                    dadzbb_mat[m][j] = \
                    np.real(np.fft.ifft(
                        A_m_coeff *
                        (kxdbjp1 * self.kxdb)
                        *sech_kh
                    ))

                    dbdzfs_mat[m][j] = \
                    np.real(np.fft.ifft(
                        B_m_coeff * 
                        (kxdbj)
                        *sech_kh
                    ))

                    dbdzdxbb_mat[m][j] = (-1)* \
                    np.real(np.fft.ifft(
                        complex(0,1) * B_m_coeff *
                        (kxdbj)
                        *(tanh_kh)
                    ))

                    dbdzbb_mat[m][j] = (-1)* \
                    np.real(np.fft.ifft(
                        B_m_coeff *
                        (kxdbjp1)
                        *(tanh_kh)
                    ))

            # B====================================

            if m < self.M - 1:
                # calculate A_{m+1}, B_{m+1}
                
                A_m_z0 = np.zeros(self.Nx)
                eta_kfact = np.copy(eta)
                fst_trm = np.zeros(self.Nx); snd_trm = np.zeros(self.Nx)
                zeta_kfact = np.ones(self.Nx)
                for i in range(m+1):
                    # A========
                    A_m_z0 -= eta_kfact * (
                        dadzfs_mat[m-i,i,:self.Nx] +
                        dbdzfs_mat[m-i,i,:self.Nx]
                    )
                    eta_kfact *= eta
                    eta_kfact /= (i+2)
                    # B========
                    fst_trm += zeta_kfact * (
                        dadzdxbb_mat[m-i,i,:self.Nx] +
                        dbdzdxbb_mat[m-i,i,:self.Nx]
                    )
                    zeta_kfact *= zeta
                    zeta_kfact /= (i+1)
                    snd_trm += zeta_kfact * (
                        dadzbb_mat[m-i,i,:self.Nx] +
                        dbdzbb_mat[m-i,i,:self.Nx]
                    )
                # Bz_m_zh = zetax*fst_trm - snd_trm
                Bz_m_zh = np.copy(zeta_x)
                Bz_m_zh *= fst_trm; Bz_m_zh -= snd_trm

                A_m_coeff = np.fft.fft(np.append(A_m_z0,
                        np.fliplr([A_m_z0])[0]))
                A_m_coeff *= self.chi #coefficient array

                B_m_coeff = np.fft.fft(np.append(Bz_m_zh,
                        np.fliplr([Bz_m_zh])[0]))
                B_m_coeff *= self.chi #coefficient array

        
        #result to return
        w = np.zeros(self.Nx)
        fact = np.array([1/math.factorial(index) for index in j_indices])
        eta_fact = eta**j_indices[:,np.newaxis]
        eta_fact *= fact[:,np.newaxis]
        for m in range(self.M):
            w += np.sum(eta_fact[:self.M - m] * (
                dadzfs_mat[m,:self.M - m,:self.Nx] +
                dbdzfs_mat[m,:self.M - m,:self.Nx]
            ),axis=0)
        return w

    def volume(self):
        """
        Calculates the volume of water at this time step. This is
        equal to the integral of eta dx, so still water has a volume
        of 0.
        """
        return np.trapz(self.eta, dx=self.dx)


    def peak_location(self):
        """
        Finds and returns the position (index * dx) of the highest eta
        value.
        """
        return self.dx * np.argmax(self.eta)

    @staticmethod
    def soliton(x0,a0,h0,Nx,dx,g = 9.81):
        """
        Returns a tuple corresponding to eta and phiS of a soliton at
        a given point in space.

        x0   -- The x coordinate of the soliton, where
                x=0 corresponds with an index of 0 in the vectorization
                of eta and phiS
        a0   -- The amplitude of the soliton
        h0   -- The water deph beneath the soliton
        Nx   -- The number of points in the vectorization of eta and phiS
        dx   -- The spatial resolution (distance between points)
        g    -- acceleration due to gravity [default: 9.81]
        """
        alpha = a0/h0
        K0 = math.sqrt((3*a0)/(4*h0**3)) \
                * (1 - (5/8)*alpha + (71/128)*alpha**2)
        
        sim_length = Nx * dx; x = np.linspace(0,sim_length,Nx)

        s2 = np.cosh(K0 * (x - x0))
        s2 **= -2
        ta = np.tanh(K0 * (x - x0))
        s2t = s2 * ta
        s4t = s2t * s2

        eta = h0*((alpha)*s2 - (3/4)*alpha**2*(s2-s2**2) + alpha**3*(
            (5/8)*s2 - (151/80)*s2**2 + (101/80)*s2**3))
        ps = (((alpha)*math.sqrt(g*h0)/(math.sqrt((3*a0)/(4*h0**3)))) \
            *(ta+(alpha)*(5/24*ta-1/3*s2t+3/4*(1+eta/h0)**2*s2t) \
            +(alpha)**2*(-1257/3200*ta+9/200*s2t+6/25*s4t+(1+eta/h0)**2* \
            (-9/32*s2t-3/2*s4t)+(1+eta/h0)**4*(-3/16*s2t+9/16*s4t))))

        return (eta,ps)
    
    @staticmethod
    def KY_bathym(Nx = 2**14, dx = 0.04, s0=0.002, d0 = 0.9,
            gamma = 0.1, X1 = 4):
        """
        Produces a bathymetry profile similar to Knowles and Yeh's paper.
        expects h0 = 1, but the result can be multiplied by the desired h0.

        Nx    - number of points
        dx    - spatial resolution (distance between each point)
        s0    - nominal slope of the bathymetry
        d0    - height of the beach plateau
        gamma - smoothing parameter
        X1    - position where the bathymetry should start sloping up
        """
        x = np.linspace(-X1,(Nx-1)*dx - X1, Nx)
        bath = np.zeros(Nx)
        
        l1 = 2*gamma*d0/(s0)
        l2 = ((1-gamma)*d0 - (s0*l1)/2)/s0 + l1
        l3 = l2 + l1

        sqr_coeff = s0/(2 * l1)
        lin_off = s0*l1 / 2 - 1

        d0 -= 1
        for (i,x) in enumerate(x):
            if x < 0:
                bath[i] = -1
            elif x < l1: # smoothed
                bath[i] = sqr_coeff * x*x - 1
            elif x < l2: # linear domain
                bath[i] = lin_off + s0 * (x - l1)
            elif x < l3: # smoothed
                bath[i] = d0 - (sqr_coeff * (x-l3)**2)
            else:
                bath[i] = d0

        return bath

    @staticmethod
    def KY_SIM(Nx=2**14, dx = 0.04, dt = 0.01, s0 = 1/500, x0=30, a0=0.1, h0=1):
        """
        Returns a new simulator similar to Knowles and Yeh's
        initial conditions.

        Nx        - Number of points (nodes) in the discreteized simulation
        dx        - Spatial resolution
        s0        - Slope of the bathymetry
        x0        - location of the center of the starting soliton
        a0        - amplitude of the soliton
        h0        - depth of the water
        """
        return Simulator1D(
            h0*Simulator1D.KY_bathym(Nx=Nx, dx=dx, s0=s0,X1 = x0*2),
            dt, dx,
            *Simulator1D.soliton(x0, a0, h0, Nx, dx)
        )
