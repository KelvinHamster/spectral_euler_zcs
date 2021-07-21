"""
Tests if the simulator properly conserves the soliton when
there are no external forces, and there is a flat bottom.
"""
from __future__ import division

#handle command line arguments
import argparse

parser = argparse.ArgumentParser(description="Tests if the simulator properly conserves the soliton when" +\
    "there are no external forces, and there is a flat bottom.")

parser.add_argument("-x","--dx", help="set the spatial resolution", type=float, default=0.04)
parser.add_argument("-t","--dt", help="set the time step", type=float, default=0.01)
parser.add_argument("-a","--a0", help="initial amplitude", type=float, default=0.1)
parser.add_argument("--h0", help="water depth", type=float, default=1.0)
parser.add_argument("-n","-N","--Nx", help="number of points/nodes", type=int, default=2**14)
parser.add_argument("-s","--step", help="number of time steps per validation step", type=int, default=10)
parser.add_argument("-d","--duration", help="set the maximum time to run", type=float, default=10)
parser.add_argument("-v","--lowpass", help="set the low-pass filter", type=float, default=0.7)
parser.add_argument("-i","--ignore", help="margin on edges of simulation to ignore when measuring offsets (in x-space)", type=float, default=25)
parser.add_argument("-m","--method", help="set the integration method", type=str, default="RK4")
parser.add_argument("-S","--save", help="if a json file containing the data should be saved or not", action="store_true")
parser.add_argument("--netcdf", help="if the data should be stored in netCDF along side eta/phi values", action="store_true")
parser.add_argument("-o","--output", help="directory/filename, no suffix, of where to save the data.", type=str, default="flat_unforced")

args = parser.parse_args()

#in case we call this from the git repo, the module will be in the parent
#directory. Add the parent directory to path, so that we see it
import sys
sys.path.append("../")

from euler_model.simulator import Simulator1D
import numpy as np
import math

import scipy.optimize as optimize
def minimize_func(f, x0):
    """
    Calculates the shift and distance in L^inf of eta and pS to the soliton.
    """
    def f2(x):
        return f(x[0])
    res = optimize.minimize(f,x0,method="Nelder-Mead")
    if res.success:
        return res.x[0]
    else:
        return None

Nx = args.Nx
dx = args.dx
dt = args.dt
h0 = args.h0
a0 = args.a0
tmax = args.duration
method = args.method
valstep = args.step
ignore_region = args.ignore

usencdf = args.netcdf


x0 = Nx*dx/2

sim = Simulator1D(np.ones(Nx)*(-h0), dt, dx,
    *Simulator1D.soliton(x0,a0,h0,Nx,dx),
v=args.lowpass)
sim.a0 = a0

measure_window = (sim.x > ignore_region)*(sim.x < (sim.sim_length - ignore_region))

last_x = x0
def measure_offset(eta, pS):
    """
    Calculates the shift and distance in L^2 of eta and pS to the soliton.
    """
    def f(x):
        e0, pS0 = Simulator1D.soliton(x,a0,h0,Nx,dx)
        return np.linalg.norm(measure_window*(e0-eta),ord=2) +\
               np.linalg.norm(measure_window*(pS0-pS),ord=2)
    res = minimize_func(f,last_x)
    if res != None:
        #return max infinity norm
        e0, pS0 = Simulator1D.soliton(res,a0,h0,Nx,dx)
        return (res, 
                np.linalg.norm(measure_window*(e0-eta),ord=np.inf),
                np.linalg.norm(measure_window*(pS0-pS),ord=np.inf))
    else:
        return (None,None,None)


measures = []

if usencdf:
    from scipy.io import netcdf
    f = sim.init_netcdf(f"{args.output}.nc", True, "zero", close = False)
    timevar = f.variables['time']
    #new variables for cool stuff
    f.createVariable('solit_x', np.float64, ('time',))
    f.createVariable('solit_eta_err', np.float64, ('time',))
    f.createVariable('solit_pS_err', np.float64, ('time',))



step = 0
while sim.t <= tmax:
    if step % valstep == 0:
        #log error values
        x, eta_err, phi_err = measure_offset(sim.eta, sim.phiS)
        frame = {
            "t":sim.t,
            "x":x,
            "eta_error":eta_err,
            "pS_error":phi_err
        }
        measures.append(frame)
        print(frame)
        if x != None:
            last_x = x
        
        #save netcdf: overwrite step 0 just because
        if usencdf:
            t_index = len(timevar[:]) if (step > 0) else 0
            timevar[t_index] = sim.t
            f.variables['eta'][t_index] = sim.eta
            f.variables['pS'][t_index] = sim.phiS
            f.variables['solit_x'][t_index] = x
            f.variables['solit_eta_err'][t_index] = eta_err
            f.variables['solit_pS_err'][t_index] = phi_err
    

    sim.step(method)
    step += 1

if usencdf:
    f.close()


if args.save:
    fname = args.output
    import json
    import os
    #set unique name so no overwriting
    if os.path.exists("%s.json" % (fname)):
        i = 0
        while os.path.exists("%s%d.json" % (fname,i)):
            i += 1
        fname += str(i)
    fname += ".json"
    
    with open(fname, "w") as f:
        json.dump({
            "meta":{
                "dx":dx,
                "dt":dt,
                "Nx":Nx,
                "h0":h0,
                "a0":a0,
                "tmax":tmax,
                "method":method,
                "steps_per_validation":valstep,
                "margin_ignored":ignore_region,
                "lowpass": args.lowpass
            },
            "data":measures
        },f)