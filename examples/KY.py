"""
Simulates Knowles and Yeh (2018)
"""

#in case we call this from the git repo, the module will be in the parent
#directory. Add the parent directory to path, so that we see it
import sys
sys.path.append("../")

from euler_model.simulator import Simulator1D

def sim_KY(Nx = 2**14, a0 = 0.1, h0 = 1.0, X0 = -280, d0 = 0.9, s0 = 1.0/500,
        dx = 0.04, dt = 0.01, Xt = 50, mass_err_crit = 0.5,
        energy_err_crit = 0.5, tmax = 500, v=0.7, M = 5, g = 9.81,
        tsave = 1, save_dir = None, gui = True):
    """
    Creates a simulation similar to Knowles and Yeh's conditions. Takes
    the following optional arguments:

    Nx              [default: 2**14]
          - number of points (nodes) in the simulation.
    
    a0              [default: 0.1]
          - amplitude of the soliton initial condition

    h0              [default: 1.0]
          - base water depth. (depth at zeta = 0)

    X0              [default: -280]
          - position (not vector index) of the soliton relative to the center
            of the simulation space. Negative values correspond to the left
            half of the simulation space.

    d0              [default: 0.9]
          - height of the beach plateau (on the right side)

    s0              [default: 1/500]
          - nominal slope of the bathymetry.

    dx              [default: 0.04]
          - spatial resolution (distance between points/nodes in simulation
            space)
    
    dt              [default: 0.01]
          - Time step of the simulation
    
    Xt              [default: 50]
          - Distance from the soliton wave crest to the beach toe. Positive
            values mean that the beach toe is in front of the wave

    mass_err_crit   [default: 0.5]
          - Largest deviation in mass (integral of eta) allowed before the
            simulation is terminated
            
    energy_err_crit [default: 0.5]
          - Largest deviation in energy allowed before the
            simulation is terminated
    
    tmax            [default: 500]
          - Largest time in simulation allowed before termination.

    v               [default: 0.7]
          - The low-pass filter. Takes a number to cut off all wavenumbers
            greater than v*max(k), or takes a function that maps k and max(k)
            to how much the corresponding amplitude should be scaled. See
            the constructor of Simulator1D.
    
    M               [default: 5]
          - The number of terms in the pertubation expansion to simulate

    g               [default: 9.81]
          - Acceleration due to gravity

    tsave           [default: 1]
          - time in between plot saves. Zero or negative corresponds to no
            saving. This value is rounded to a multiple of dt.
    
    save_dir        [default: "./KY_dx[dx]_dt[dt]_s[s0]_a[a0]_plot"]
          - the directory/file prefix for the saved plots.
    """
    if gui:
        import matplotlib.pyplot as plt
    #initial conditions:

    #in KY, x=0 is the middle, change it to the left
    sim_len = Nx * dx
    X0 += sim_len/2

    bathym = Simulator1D.KY_bathym(Nx,dx,s0,d0, X1=Xt + X0) * h0
    init_cond = Simulator1D.soliton(X0,a0,h0,Nx,dx,g)

    sim = Simulator1D(bathym, dt, dx, *init_cond, M=M, v=v, g=g, h0=h0)
    
    if save_dir == None:
        #save_dir = f"./KY_dx{dx}_dt{dt}_s{s0}_a{a0}_plot"
        save_dir = "./KY_dx"+str(dx)+"_dt"+str(dt)+"_s" \
            +str(s0)+"_a"+str(a0)+"_plot"

    if gui:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    def on_loop(sim, step, plot, data):
        if gui:
            ax.clear()
            ax.plot(sim.x, sim.eta, "b")
            ax.plot(sim.x, sim.zeta - sim.h0, "k")
            ax.set_ylabel("z")
            ax.set_xlabel("x")
            #ax.set_title(f"dx={sim.dx},dt={sim.dt},t={round(sim.t,3)}")
            ax.set_title("dx="+str(sim.dx)+",dt="+str(sim.dt)+ \
                ",t="+str(round(sim.t,3)))
            plt.pause(0.05)
        else:
            if step % 100 == 0:
                print("Time: "+sim.time)
    
    def continue_test(s):
        if abs(s.volume() - base_mass) > mass_err_crit:
            print("Critical volume error reached! Stopping!")
            return False
        elif abs(s.energy() - base_energy) > energy_err_crit:
            print(s.energy(), base_energy, abs(s.energy() - base_energy), energy_err_crit)
            print("Critical energy error reached! Stopping!")
            return False
        elif s.t > tmax:
            print("Maximum time reached! Stopping!")
            return False
        return True

    if gui:
        fig.show(False)


    def on_plotsave(sim, filename):
        if gui:
            fig.savefig(filename)

    base_mass = sim.volume()
    base_energy = sim.energy()

    sim.run_simulation(tsave,0,save_dir,
        continue_test,
        integrator = "RK4",
        loop_callback = on_loop,
        plot_func=on_plotsave
    )
    if gui:
        plt.close(fig)


#only call everything in the if statement if we ran KY and not imported it
if __name__ == '__main__':
    sim_KY()