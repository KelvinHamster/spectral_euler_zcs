import sys
sys.path.append("../")

from euler_model.simulator import Simulator1D

import ffmpeg
#handle command line arguments
import argparse

parser = argparse.ArgumentParser(description="Tests if the simulator properly conserves the soliton when" +\
    "there are no external forces, and there is a flat bottom.")

parser.add_argument("-x","--dx", help="set the spatial resolution", type=float, default=0.1)
parser.add_argument("-t","--dt", help="set the time step", type=float, default=0.001)
parser.add_argument("-a","--a0", help="initial amplitude", type=float, default=0.1)
parser.add_argument("-s","--s0", help="slope of the bathymetry", type=float, default=0.0)
parser.add_argument("-P","--wind", help="wind pressure coefficient", type=float, default=0.0)
parser.add_argument("--X0", help="position of the soliton initial condition", type=float, default=120.0)
parser.add_argument("--X1", help="position of the beach toe", type=float, default=170.0)
parser.add_argument("--h0", help="water depth", type=float, default=1.0)
parser.add_argument("-N","--Nx", help="number of points/nodes", type=int, default=2**13)
parser.add_argument("--plotstep", help="amount of time between each plot", type=float, default=0.0)
parser.add_argument("--plotspect", help="whether or not the spectral data of the simulation should be plotted", action="store_true")
parser.add_argument("--savestep", help="amount of time between each saving of a data frame", type=float, default=0.5)
parser.add_argument("-d","--duration", help="set the maximum time to run", type=float, default=10)
parser.add_argument("-v","--lowpass", help="set the low-pass filter cut-off", type=float, default=0.7)
parser.add_argument("-A","--lowpass_speed", help="set how fast the low-pass filter transitions", type=float, default=1.0)
parser.add_argument("-m","--method", help="set the integration method", type=str, default="RK4")
parser.add_argument("--netcdf", help="if the data should be stored in netCDF along side eta/phi values", action="store_true")
parser.add_argument("-o","--output", help="directory/filename, no suffix, of where to save the data.", type=str, default="flat_unforced")

args = parser.parse_args()

slope = args.s0
wind_coeff = args.wind
plot_spectrum = args.plotspect
h0 = args.h0
savencdf = args.netcdf

timestop = args.duration

data_dir = args.output


import numpy as np

v = args.lowpass
A = args.lowpass_speed

def lowpass(k,peak):
    return (1 - np.tanh(A*(k - v*peak)))/2
# sim = Simulator1D.KY_SIM(s0 = slope,x0=40)


dx = args.dx
dt = args.dt
Nx = args.Nx

a0 = args.a0
bathym = None
if slope == 0:
    bathym = np.zeros(Nx)-1
else:
    bathym = Simulator1D.KY_bathym(Nx = Nx, dx = dx, s0 = slope, X1 = args.X1) * h0

sim = Simulator1D(bathym,dt,dx,
    *Simulator1D.soliton(args.X0,a0,h0,Nx,dx),
    v=lowpass, P=wind_coeff)
sim.a0 = a0


plot_dt = args.plotstep
base_volume = sim.volume()

import matplotlib.pyplot as plt
def stop_cond(sim):
    #max time
    #volume conservation
    #froude number stop at 0.8
    return sim.t <= timestop and \
            abs(sim.volume() - base_volume) < 0.005 * sim.sim_length and\
            (sim.max_v / np.sqrt(9.81)) <= 0.8
def callback(sim,step,shouldplot,shoulddata):
    if step % 100 == 0:
        print(f"Time: {round(sim.t,3)}, Volume difference: {abs(sim.volume() - base_volume)}"\
            + f", Fr: {sim.max_v/np.sqrt(9.81)}")
    if shouldplot and plot_spectrum:
        KE = (sim.phiS * sim.calculate_time_derivatives(
                sim.eta, sim.phiS, sim.zeta, sim.zeta_x,
                np.zeros(Nx), np.zeros(Nx)
        )[0])
        PE = sim.g * sim.eta**2
        fftKE = np.fft.fft(np.append(KE, np.fliplr([KE])[0]))
        fftPE = np.fft.fft(np.append(PE, np.fliplr([PE])[0]))
        fftphiS = np.fft.fft(np.append(sim.phiS, np.fliplr([sim.phiS])[0]))
        plt.plot(sim.kxdb, 
            np.real(fftphiS * np.conj(fftphiS))
        , "b")
        plt.savefig(f"{data_dir}phiS_spect{round(sim.t/plot_dt)}.png")
        plt.clf()

        plt.plot(sim.kxdb, 
            np.real(fftKE * np.conj(fftKE))
        , "b")
        plt.savefig(f"{data_dir}KE_spect{round(sim.t/plot_dt)}.png")
        plt.clf()
        
        plt.plot(sim.kxdb, 
            np.real(fftPE * np.conj(fftPE))
        , "b")
        plt.savefig(f"{data_dir}PE_spect{round(sim.t/plot_dt)}.png")
        plt.clf()

        fftKE += fftPE
        plt.plot(sim.kxdb, 
            np.real(fftKE * np.conj(fftKE))
        , "b")
        plt.savefig(f"{data_dir}E_spect{round(sim.t/plot_dt)}.png")
        plt.clf()

sim.run_simulation(plot_dt,args.savestep,data_dir,stop_cond,
    loop_callback=callback, integrator = args.method, save_netcdf=savencdf)

import os
try:
    if os.path.exists(f'{data_dir}0.png'):
        (
            ffmpeg
            .input(f'{data_dir}%d.png', framerate=1/plot_dt)
            .output(f'{data_dir}.mp4')
            .run()
        )
        if plot_spectrum:
            (
                ffmpeg
                .input(f'{data_dir}E_spect%d.png', framerate=1/plot_dt)
                .output(f'{data_dir}E_spect.mp4')
                .run()
            )
            (
                ffmpeg
                .input(f'{data_dir}KE_spect%d.png', framerate=1/plot_dt)
                .output(f'{data_dir}KE_spect.mp4')
                .run()
            )
            (
                ffmpeg
                .input(f'{data_dir}PE_spect%d.png', framerate=1/plot_dt)
                .output(f'{data_dir}PE_spect.mp4')
                .run()
            )
            (
                ffmpeg
                .input(f'{data_dir}phiS_spect%d.png', framerate=1/plot_dt)
                .output(f'{data_dir}phiS_spect.mp4')
                .run()
            )
except:
    pass
i=0
while os.path.exists(f'{data_dir}{i}.png'):
    os.remove(f'{data_dir}{i}.png')
    if os.path.exists(f'{data_dir}phiS_spect{i}.png'):
        os.remove(f'{data_dir}phiS_spect{i}.png')
    if os.path.exists(f'{data_dir}PE_spect{i}.png'):
        os.remove(f'{data_dir}PE_spect{i}.png')
    if os.path.exists(f'{data_dir}KE_spect{i}.png'):
        os.remove(f'{data_dir}KE_spect{i}.png')
    if os.path.exists(f'{data_dir}E_spect{i}.png'):
        os.remove(f'{data_dir}E_spect{i}.png')
    i+=1