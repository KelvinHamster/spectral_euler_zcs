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
parser.add_argument("-S","--save", help="if a file containing the data should be saved or not", action="store_true")
parser.add_argument("-o","--output", help="directory/filename, no suffix, of where to save the data.", type=str, default="flat_unforced")

args = parser.parse_args()

#in case we call this from the git repo, the module will be in the parent
#directory. Add the parent directory to path, so that we see it
import sys
sys.path.append("../")

from euler_model.simulator import Simulator1D
import numpy as np
import math

minimize_func = None
try:
    import scipy.optimize as optimize
    def minimize(f, x0):
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
    minimize_func = minimize
except:
    def approx_grad(f,x, h0=0.0001, n = 3):
        """
        approximates f' at x using Richardson Extrapolation off of the central
        difference formula.
        """
        def N(i,h):
            #base case: i=1
            if i==1:
                return (f(x+h) - f(x-h))/(2*h)
            elif i > 1:
                i -= 1
                return (2**(2*i)*N(i,h/2) - N(i,h))/(2**(2*i) - 1)
            return 0
        return N(n,h0)

    def bfgs_step(f, xk, Bk, eps = 0.0001):
        #search direction
        fp = approx_grad(f,xk)
        p = -fp/Bk
        if abs(p) > 1:
            p /= abs(p)*2
        fk = f(xk)
        #line search / armijo condition
        armijo = 0.25
        alpha = 1
        while f(xk + alpha*p) > fk + armijo * (fp*alpha*p) and alpha > eps:
            alpha /= 2
        d = alpha*p
        xkp1 = xk + d
        y = approx_grad(f,xkp1) - fp
        #bfgs for Bkp1
        Bkp1 = Bk
        if abs(y*d) > eps**4 and abs(d*Bk*d) > eps**6:
            Bkp1 += y*y/(y*d) - Bk*d*d*Bk/(d*Bk*d)
        return (xkp1,Bkp1,fp)

    def minimize(f, x0, eps = 0.00001):
        #approx hessian/2nd deriv with finite difference
        B0 = (f(x0+eps) - 2*f(x0) + f(x0-eps))/(eps*eps)
        for i in range(500):
            #force to gradient descent if there is negative curvature
            recalc_B = False
            if B0 < eps:
                B0 = 1
                recalc_B = True
            x1, B0, fp = bfgs_step(f,x0,B0)
            if abs(fp) < eps or abs(x1-x0) < eps*eps:
                break
            x0 = x1
            if recalc_B:
                B0 = (f(x0+eps) - 2*f(x0) + f(x0-eps))/(eps*eps)

        return x1
    minimize_func = minimize

Nx = args.Nx
dx = args.dx
dt = args.dt
h0 = args.h0
a0 = args.a0
tmax = args.duration
method = args.method
valstep = args.step
ignore_region = args.ignore


x0 = Nx*dx/2

sim = Simulator1D(np.ones(Nx)*(-h0), dt, dx,
    *Simulator1D.soliton(x0,a0,h0,Nx,dx),
v=args.lowpass)

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

step = 0
while sim.t <= tmax:
    if step % valstep == 0:
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
    sim.step(method)
    step += 1

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