from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import netcdf

from euler_model.integrator import Integrator1D

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
            v    -- lowpass threshold. If v is a number, any wavenumber greater
                    than the largest wavenumber times v is clipped off each
                    timestep. v can also be a function that takes a wavenumber
                    and the peak wavenumber, and returns how it should be scaled
                    [default: 0.7]
            g    -- acceleration due to gravity. [default: 9.81]
            h0   -- base still water depth [default: bathymetry[0]]
        """
        self.dt = dt; self.dx = dx
        self.eta = eta0; self.phiS = phiS0
        
        self.M = M; self.g = g
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
        k_peak = max(self.kappadb)
        if callable(v):
            self.chi = np.array([v(k,k_peak) for k in self.kappadb])
        else:
            self.chi = self.kappadb <= (v*k_peak)
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

    def step(self, method, P_atmos = None, *args, **kwargs):
        """
        Steps the simulation forward using the given method. Any arguments
        for the method can be specified as optional arguments or keyword
        arguments.

        method
              - The method to use. This can be a string or a function. When
                a string is passed the method in integrator.py of the same
                name is used.

        P_atmos
              - Atmospheric pressure at the surface.
        """
        if callable(method):
            method(self,P_atmos,*args,**kwargs)
        else:
            getattr(Integrator1D, method)(self,P_atmos,*args,**kwargs)

    def calculate_gradient(self, vec):
        """
        Takes a vectorized surface vec, and calculates its spatial derivative
        using an FFT trick.

        Calling calculate_gradient(eta) returns eta_x
        """
        fourier_vec = np.fft.fft(np.append(vec, np.fliplr([vec])[0]))
        fourier_vec *= self.kxdb_im
        vec_x = np.real(np.fft.ifft(fourier_vec))[0:self.Nx]
        return vec_x

    def diff_eval(self, eta, phiS_x, eta_x, w, P_a):
        """
        Evaluates the differential equation based on the given parameters.
        Returns a tuple (eta_t, phiS_t) of the time derivatives.

        phiS_x    - gradient of phiS
        eta_x     - gradient of eta
        w         - vertical velocity at the surface
        P_a       - atomspheric pressure at the surface
        """
        eta_x_sq_p1 = eta_x**2
        eta_x_sq_p1 += 1
        # not sure if these are in-place operations;
        # if not, this can be optimized.
        return (
            -phiS_x*eta_x + eta_x_sq_p1*w,
            -P_a - self.g*eta - (phiS_x**2)/2 + eta_x_sq_p1/2*(w**2)
        )

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
        P_a      -- atmospheric pressure at every point, should have the
                    same samples as bathymetry. Expected to be a numpy
                    array or a function of
                    (eta,phiS,eta_x,phiS_x,w) that returns a numpy array.
        """
        #gradients:
        eta_x = self.calculate_gradient(eta)
        phiS_x = self.calculate_gradient(phiS)

        w = self.vertvel(eta, phiS, zeta, zeta_x, zeta_t)

        if callable(P_a):
            return self.diff_eval(eta, phiS_x, eta_x, w,
                P_a(eta,phiS,eta_x,phiS_x,w))
        else:
            return self.diff_eval(eta, phiS_x, eta_x, w, P_a)


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
        kxdbh = self.kxdb * self.h0 * self.chi
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

    def energy(self):
        """
        Calculates the energy in the wave at this time step.
        """
        eta_t,_ = self.calculate_time_derivatives(
            self.eta, self.phiS, self.zeta, self.zeta_x,
            np.zeros(self.Nx), np.zeros(self.Nx)
        )
        KE = self.phiS * eta_t
        PE = self.g * (self.eta**2)

        return np.trapz(KE + PE, dx=self.dx)

    def peak_location(self):
        """
        Finds and returns the position (index * dx) of the highest eta
        value.
        """
        return self.dx * np.argmax(self.eta)

    def zeta_at(self, x):
        """
        Finds the value of zeta at a given x position. If x is not a
        multiple of dx, zeta is evaluated from a linear interpolation.

        x
              - a number to evaluate zeta at.
        """
        i = math.floor(x/self.dx)
        i = max(0,min(i, self.Nx - 2))

        z0 = self.zeta[i]
        z1 = self.zeta[i+1]

        d = x/self.dx - i
        return z0 + (z1-z0)*d

    def run_simulation(self, saveplot_dt, savedata_dt, directory,
            should_continue = lambda sim: sim.t > 500,
            integrator = "RK4",
            save_eta = None,
            save_phi = None,
            loop_callback = None,
            plot_func = None,
            save_json = False,
            save_netcdf = True, save_buffer = 10,
            cdf_h_invariant = True,
            cdf_Pderiv = "zero", cdf_timeunits = "seconds",
            cdf_spaceunits = "h0*meters"):
        """
        Automatically integrates eta and phiS over a set of timesteps,
        saving plots and/or data at given intervals in time.

        saveplot_dt
                  - the timestep between saved plots. This number is rounded
                    to the nearest multiple of the simulation dt. If this
                    value does not round to a positive number, plots are not
                    saved. Plots are saved as PNG files with a name
                    corresponding to the order it is saved in. A plot with
                    number 'i' represents the data at time 'i*saveplot_dt'

        savedata_dt
                  - the timestep between saved data. This number is rounded
                    to the nearest multiple of the simulation dt. If this
                    value does not round to a positive number, data is not
                    saved. Data is saved as a json file with name 'dat.json'
                    which is created regardless if data should be saved or not.
                    In the case that data is not saved, only the metadata of
                    the simulation is stored.
        
        directory
                  - the directory to save the files to. This can also include
                    a prefix to the file. If directory="~/sim/", then plots are
                    saved as "[number].png" in the ~/sim/ directory. If
                    directory="~/sim", then plots are saved as
                    "sim[number].png" in the home directory. If none, no files
                    are saved.
        
        should_continue
                  - function that determines if a simulation should stop or
                    not. This takes the simulation as an argument and returns a
                    boolean. The simulation is run until should_continue
                    returns false. By default, this is the lambda function
                    sim => sim.t > 500
        
        integrator
                  - function or string that timesteps the simulation.
                    functions should only take the simulation as an argument
                    and return nothing, modifying the passed simulation.
                    Strings should be the name of a method in Integrator1D,
                    which will be called by the simulation.
                    By default, this value is "RK4"

        save_eta
                  - Parameters for how eta should be saved when data is saved
                    to json. This should be generated using
                    Simulator1D.data_save_params(). If None, then eta is not
                    saved.
        
        save_phi
                  - Parameters for how phiS should be saved when data is saved
                    to json. This should be generated using
                    Simulator1D.data_save_params(). If None, then phiS is not
                    saved.

        loop_callback
                  - this function is called after every simulation step. It
                    should be a void function that takes the arguments
                    sim, step, plot, data, where
                    <sim> is the simulator at the step
                    <step> is the integer multiple of dt that the simulation
                    has run
                    <plot> is a boolean representing if the plot was saved this
                    step
                    <data> is a boolean representing if the data was saved this
                    step
                    
                    By default, loop_callback makes a print statement after
                    every 100 time steps.
        plot_func
                  - A function that is dedicated to plotting and saving the
                    figure. The function is expected to be void and take the
                    arguments (sim, filename).

        save_json
                  - Whether or not to save the file to json. The metadata of
                    the simulation is saved even if savedata_dt is not
                    positive. Setting save_json to false prevents this.
        save_netcdf
                  - Whether or not to save the file to netcdf. The netCDF file
                    ignores data truncation specifications of save_eta and
                    save_phi.
        save_buffer
                  - number of datapoints to buffer in between file-writes. If 0
                    or 1, then every savedata_dt, the json/netCDF file is opened
                    and written to.
        cdf_h_invariant
                  - Whether or not h is treated as invariant. If false, then
                    the bathymetry is saved every frame, alongside eta and
                    phiS.
        cdf_Pderiv
                  - The string that should populate the P_deriv field in the
                    netcdf file.
        cdf_timeunits
                  - The string that specifies the units of time for the
                    simulation.
        cdf_spaceunits
                  - The string that specifies the units of x for the
                    simulation.
        """
        if not callable(integrator):
            method = integrator
            integrator = lambda sim: sim.step(method)

        if loop_callback == None:
            def cb(sim, step, plot, data):
                if step % 100 == 0:
                    #print(f"Time: {round(sim.t,3)}")
                    print("Time: "+str(round(sim.t,3)))
            loop_callback = cb

        if save_netcdf:
            ncdf_filename = f"{directory}sim.nc"
            self.init_netcdf(ncdf_filename, cdf_h_invariant,
                    cdf_Pderiv, cdf_timeunits, cdf_spaceunits, close=True)
        #data for method below:
        netcdf_buffer_t = []
        netcdf_buffer_eta = []
        netcdf_buffer_pS = []
        netcdf_buffer_h = []
        #call whenever we want to put the next timestep in the netcdf file
        def append_netcdf(sim, step, flush_only = False):
            if not save_netcdf:
                return
            #append the data to the buffer
            if not flush_only:
                netcdf_buffer_t.append(sim.t)
                netcdf_buffer_eta.append(sim.eta)
                netcdf_buffer_pS.append(sim.phiS)
                if cdf_h_invariant == 0:
                    netcdf_buffer_h.append(sim.h0 - sim.zeta)
            #if the buffer is full or we force a flush, write it out
            buf_size = len(netcdf_buffer_t)
            if buf_size >= save_buffer or (flush_only and buf_size > 0):
                f = netcdf.netcdf_file(ncdf_filename, 'a')
                time = f.variables['time']
                for j in range(buf_size):
                    i = len(time[:])
                    time[i] = netcdf_buffer_t[j]
                    
                    #copy values
                    f.variables['eta'][i] = netcdf_buffer_eta[j]
                    f.variables['pS'][i] = netcdf_buffer_pS[j]
                    # set bathymetry if variant
                    if cdf_h_invariant == 0:
                        f.variables['h'][i] = netcdf_buffer_h[j]
                del time
                f.close()
                netcdf_buffer_t.clear()
                netcdf_buffer_eta.clear()
                netcdf_buffer_pS.clear()
                netcdf_buffer_h.clear()

        step = 0

        #how many steps per plot
        plotstep = round(saveplot_dt/self.dt)
        saveplot = plotstep > 0

        #how many steps per data save
        datastep = round(savedata_dt/self.dt)
        savedata = datastep > 0
        
        if plot_func == None and saveplot:
            import matplotlib.pyplot as plt
            def do_plot(sim, filename):
                plt.plot(sim.x, sim.eta, "b")
                plt.plot(sim.x, sim.zeta - sim.h0, "k")
                plt.ylabel("z")
                plt.xlabel("x")
                plt.title(f"dx={sim.dx},dt={sim.dt},t={round(sim.t,3)}")
                plt.savefig(filename)
                plt.clf()
            plot_func = do_plot
        
        
        #prep data and write metadata
        meta = {
            "start_time": self.t,
            "dt": self.dt,
            "dx": self.dx,
            "g": self.g,
            "h0": self.h0
        }
        d = {}
        import json
        if saveplot:
            meta["plot_dt"] = plotstep * self.dt
        if savedata:
            meta["savedata"] = {
                "dt": datastep * self.dt,
                "eta": save_eta,
                "phiS": save_phi
            }
            if save_phi != None:
                d["phiS"] = []
            if save_eta != None:
                d["eta"] = []
        
        while should_continue(self):
            #record data
            shouldplot = saveplot and (step % plotstep == 0)
            shoulddata = savedata and (step % datastep == 0)
            
            if shouldplot and directory != None:
                plot_func(self, f"{directory}{step//plotstep}.png")
            if shoulddata:
                if save_eta != None:
                    d["eta"].append(
                        Simulator1D.vec_to_data(self.eta, self.dx, save_eta))
                if save_phi != None:
                    d["phiS"].append(
                        Simulator1D.vec_to_data(self.phiS, self.dx, save_phi))
                
                #netcdf data: we already wrote the first data point
                if step > 0:
                    append_netcdf(self, step)

                #save the file after every 10 data collections so we don't lose
                #much data when we stop
                if (step//datastep) % save_buffer == 0 and directory != None:
                    meta["datapoints"] = len(d["eta"]) if "eta" in d \
                            else (len(d["phiS"]) if "phiS" in d else 0)
                    data = {"meta":meta, "data":d}
                    
                    with open(f"{directory}dat.json","w") as f:
                        json.dump(data, f)
            loop_callback(self, step, shouldplot, shoulddata)
            #step forward
            integrator(self)
            step += 1
        
        meta["datapoints"] = len(d["eta"]) if "eta" in d \
                else (len(d["phiS"]) if "phiS" in d else 0)
        data = {"meta":meta, "data":d}
        
        append_netcdf(self, step, flush_only=True)
        if directory != None:
            with open(f"{directory}dat.json","w") as f:
                json.dump(data, f)

    def init_netcdf(self, filename, h_invariant, P_deriv,
            timeunits = "seconds", spaceunits = "h0*meters",
            P = 0, close = True):
        """
        Generates a netCDF file of the given filename and populates it with
        one point in time representing the simulation's current state.
        Returns the netCDF_File object.

        filename
                  - The name of the file to be saved. Overwrites existing files

        h_invariant
                  - Whether the simulation should be treated as if h does not
                    vary with time
        
        P_deriv
                  - Information on how pressure is obtained. Expects "zero",
                    "wind" or "custom".
        
        timeunits
                  - A string representing the units for time
        
        spaceunits
                  - A string representing the units for spatial coordinates

        P
                  - if P_deriv is "zero" then this does nothing.
                    If "wind", then the P attribute is set to this value.
                    If "custom", then P is the P_a variable at time index 0.

        close
                  - Whether or not this method should close the netcdf file
                    resource after initialization.
        """
        # initialize the file
        f = netcdf.netcdf_file(filename, 'w')
        f.dx = np.array((self.dx),dtype=np.float64)
        f.dt = np.array((self.dt),dtype=np.float64)
        f.Nx = self.Nx
        if hasattr(self,'a0'):
            f.a0 = self.a0
        f.h0 = self.h0
        f.M = self.M
        f.g = np.array((self.g),dtype=np.float64)
        f.length = np.array((self.Nx*self.dx),dtype=np.float64)
        f.h_invariant = h_invariant
        f.P_deriv = P_deriv

        f.createDimension('time', None)
        f.createDimension('x', self.Nx)
        time = f.createVariable('time', np.float64, ('time',))
        time.units = timeunits
        time[0] = self.t
        x = f.createVariable('x', np.float64, ('x',))
        x.units = spaceunits
        x[:] = self.x

        if h_invariant:
            h = f.createVariable('h', np.float64, ('x',))
            h[:] = self.h0 - self.zeta
        else:
            f.createVariable('h', np.float64, ('time','x'))
            h[0] = self.h0 - self.zeta
        eta = f.createVariable('eta', np.float64, ('time','x'))
        eta[0] = self.eta
        ps = f.createVariable('pS', np.float64, ('time','x'))
        ps[0] = self.phiS
        chi = f.createVariable('chi', np.float64, ('x',))
        chi[:] = self.chi[:self.Nx]
        if close:
            f.close()
        return f


    @staticmethod
    def vec_to_data(vec, dx, params):
        """
        Converts a vector into data, specified by params, which is in the
        form generated by Simulator1D.data_save_params(). dx is the spatial
        resolution of vec
        """
        eps = params["eps"]
        z_trunc = params["zero_trunc"]
        mod_eps = None
        if eps > 0:
            mod_eps = lambda x: \
                    round((x if abs(x) > z_trunc else 0)/eps)*eps
        else:
            mod_eps = lambda x: \
                    x if abs(x) > z_trunc else 0
        step = 1
        if params["dx"] != None:
            step = max(round(params["dx"]/dx),1)


        if params["point_conversion"] == True:
            lin_tol = params["lin_tol"]
            #to points: should return an array of points
            pts = [ [0,mod_eps(vec[0])] ]
            #always save first and last points, skip them in loop
            for i in range(step, len(vec), step)[:-1]:
                #check linearity: (i,y) fit in linear model of (i0,y0)->(i1,y1)
                y = mod_eps(vec[i])
                y0 = pts[-1][1]
                i0 = pts[-1][0]/dx
                i1 = i+step
                y1 = mod_eps(vec[i1])
                if abs(
                    (y-y0) - (y1-y0)*(i - i0)/(i1 - i0)
                ) > lin_tol:
                    pts.append([i*dx, y])
            pts.append([ (len(vec)-1)*dx, mod_eps(vec[-1]) ])
            return pts
        else:
            pts = [
                mod_eps(y) for y in (vec[::step])[:-1]
            ]
            pts.append(mod_eps(vec[-1]))
            return pts


    @staticmethod
    def data_save_params(dx = None, point_conversion = False, eps = 0,
            lin_tol = -1, zero_trunc = 0):
        """
        Returns a dictionary of parameters for how to save data from a
        simulation. The output of data_save_params() should be used for
        arguments save_eta and save_phi in run_simulation().

        dx    
              - The spatial resolution to save with. If None, then the
                resolution is the same as the simulation. This value
                will always be rounded to a whole number multiple of
                the simulaton dx. [default: None]

        point_conversion
              - A boolean that represents if data should be coded as
                a vector (array), or if the vector should be converted
                into a list of (x,y) points. If true, then the conversion
                is made. [default: False]

        eps
              - The tolerance of the save data. The data is rounded to the 
                nearest multiple of eps. That is, with eps=0.001, the data
                is saved up to the 3rd decimal place. 0 corresponds with
                no rounding. [default: 0]
        
        lin_tol
              - Only used when point_conversion is true. Specifies a tolerance
                for which points should not be saved when they are close enough
                to a linear interpolation of the data. If the points are 
                {(0,0),(0.5,0.5),(1,1)}, any nonnegative tolerance will discard
                (0.5,0.5). If no points should be discarded, a negative value
                should be given. [default: -1]
        
        zero_trunc
              - If a value is less than this distance from 0, the value is 
                truncated to 0 before saving. [default: 0]
        """
        opt = {
            'dx': dx, 'point_conversion':point_conversion,
            'eps':eps, 'zero_trunc':zero_trunc
        }
        if point_conversion:
            opt['lin_tol'] = lin_tol
        return opt

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
    def solitonFF(x0,a0,h0,Nx,dx,g = 9.81):
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
    def smooth_bathy(h, i, Lwidth, dx):
        """ 
        smooths out the bathy
        """
        nwidth = round(Lwidth/dx)
        K = np.arange(-nwidth, nwidth+1)
        W = np.exp( -(K*dx/Lwidth)**2 )
        WTOT = np.sum(W)
        hnew = np.sum( h[i +K]* W)/WTOT
        return hnew



    @staticmethod
    def KY_bathyFF(L = 500.0, dx = 0.04, h0 = 1, s0=0.002, d0 = 0.1,
                   gamma = 0.1, Ltoe = 100.0):
        """
        Produces a FF version of bathymetry profile similar to Knowles and Yeh's paper.
        expects h0 = 1, but the result can be multiplied by the desired h0.

        L    -  distance of domain in meters: offshore is x=0  onshore is x=L
        dx    - spatial resolution (distance between each point in meters)
        h0    - deep water depth in meters
        s0    - nominal slope of the bathymetry
        d0    - height of the beach plateau in meters
        gamma - smoothing parameter
        Ltoe  - position where the bathymetry should start sloping up
        """
        x = np.arange(0, L+dx, dx)
        xtrue = x
        xtoe = x-Ltoe
        Nx = x.size
        h  = np.zeros(Nx)

        h = h0-xtoe*s0      # basic slope bathy
        h2 = h
        Lwidth=5  # this is hard coded in 5m smoothing scale.  

        for (i,xnew) in enumerate(x): # make the offshore flat and onshore flat regions
            if xnew < Ltoe:
               h[i] = h0;
            elif h[i] < d0:
               h[i]=d0;

        # now smooth out the entire domain
        for (i,xnew) in enumerate(x):
             if xnew > Lwidth*2.0 and  xnew < L-2*Lwidth:
                h2[i] = Simulator1D.smooth_bathy(h,i,Lwidth,dx) 

        return xtrue, h



    @staticmethod
    def KY_SIM(Nx=2**14, dx = 0.04, dt = 0.01, s0 = 1.0/500,
            x0=30, a0=0.1, h0=1):
        """
        Returns a new simulator similar to Knowles and Yeh's
        initial conditions.

        Nx        - Number of points (nodes) in the discreteized simulation
        dx        - Spatial resolution
        dt        - time step
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
