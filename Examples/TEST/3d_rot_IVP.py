#Dedalus3
"""
Dedalus script for 3D Anelastic Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.
"""

# imports for a bunch of things
import numpy as np
from mpi4py import MPI
import time
import sys 
import h5py


from dedalus import public as d3
from dedalus.extras import flow_tools
import pathlib
 
import logging
logger = logging.getLogger(__name__)

# Imports a parameter file "run_param_file.py"
import run_param_file as rpf   

# create data storage directory
save_direc = "raw_data/"
pathlib.Path(save_direc).mkdir(parents=True, exist_ok=True)

#if it is not a fresh simulation read in the checkpoint file name
fresh_simulation = rpf.fresh_simulation
if not fresh_simulation:
     restart_file = "raw_data/checkpoints/" + rpf.restart_file	+ ".h5"


debug_tau_terms = 'True'

#########################################################
# Model Parameters loaded from parameter file
Lx, Ly, Lz = rpf.Lx, rpf.Ly, rpf.Lz
Nx, Ny, Nz = rpf.Nx, rpf.Ny, rpf.Nz
Pr = rpf.Pr
Ra = rpf.Ra
Np = rpf.Np 
Lat = rpf.latitude
Ta = rpf.Ta
m = rpf.m
theta = rpf.theta

Q_amp = rpf.Q_amp

volume = Lx*Ly*Lz
area = Lx*Ly

T12  = Ta**(1/2)
RaPr = Ra/Pr
twothirds = 2/3

# Amplitude cutoff for NCC coefficient expansions (matches example scripts' default)
ncc_cutoff = 1e-8


#########################################################
# set up the bases
coords = d3.CartesianCoordinates('x', 'y', 'z')	
dist = d3.Distributor(coords, dtype=np.float64, mesh=[rpf.meshx,rpf.meshy] )
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=3/2)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=3/2)


#########################################################
# now we need to make a bunch of fields for our problem
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
S = dist.Field(name='S', bases=(xbasis,ybasis,zbasis))
L_buoy = dist.Field(name='L_buoy', bases=(xbasis,ybasis,zbasis))
L_diss = dist.Field(name='L_diss', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))


#########################################################
#now we want tau terms
tau_p  = dist.Field(name='tau_p')
tau_S1 = dist.Field(name='tau_S1', bases=(xbasis,ybasis))
tau_S2 = dist.Field(name='tau_S2', bases=(xbasis,ybasis))
tau_Lb = dist.Field(name='tau_Lb', bases=(xbasis,ybasis))
tau_Ld = dist.Field(name='tau_Ld', bases=(xbasis,ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))


#########################################################
# x y z gives the vectors
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)


#########################################################
# creates appropriate unit vectors
ex = dist.VectorField(coords, name='ex')
ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ey['g'][1] = 1
ez['g'][2] = 1


#########################################################
# some constants
Pr2thRa = dist.Field(name='Pr2thRa')
Pr2thRa['g'] = (Pr*Pr*theta) / Ra
Prth = dist.Field(name='Prth')
Prth['g'] = Pr*theta


#########################################################
#substitutions
lift_basis = zbasis.derivative_basis(1)
#lift_basis = zbasis.clone_with(a=1/2, b=1/2) 
lift = lambda A,n: d3.Lift(A, lift_basis, n)
div = lambda A: d3.Divergence(A)
grad = lambda A: d3.Gradient(A, coords)
cross = lambda A, B: d3.CrossProduct(A, B)
trans = lambda A: d3.TransposeComponents(A)
dz = lambda A: d3.Differentiate(A, coords['z'])
vol_avg = lambda A: d3.Integrate(A, ("x", "y", "z")) / volume
hor_avg = lambda A: d3.Integrate(A, ("x", "y")) / area
trace = lambda A: d3.trace(A)


#########################################################
# Non-constant coeffiecents
rho_ref = dist.Field(name='rho_ref', bases=zbasis)
rho_ref['g'] = (1-theta*z)**m

T_ref = dist.Field(name='T_ref', bases=zbasis)
T_ref['g'] = 1-theta*z


#########################################################
# Entropy source profile Q(z).
# Defined here as a function of z only. Not yet added to the entropy
# equation -- that wiring comes later. Placeholder profile: sin(z).
Q_w   = 0.05                          # half-width of each bump
Q_z0b = 0.075                          # bottom (heating) centre
Q_z0t = 0.925                          # top (cooling) centre

t_bot = np.abs((z - Q_z0b)/Q_w)
t_top = np.abs((z - Q_z0t)/Q_w)
Q_bot = np.where(t_bot <= 1, (1 - t_bot)**3, 0.0) * (2.0/Q_w)   # +, integ=+1
Q_top = np.where(t_top <= 1, (1 - t_top)**3, 0.0) * (2.0/Q_w)   # magnitude, integ=+1

Q = dist.Field(name='Q', bases=zbasis)
Q['g'] = Q_amp * (Q_bot - Q_top)     # heating at bottom minus cooling at top



#########################################################
# NCC bandwidth diagnostic (as in the example scripts): reports how many
# coefficients in each reference-state product survive the cutoff. Broadband
# NCCs (many surviving coefficients) are the expensive ones.
logger.info("NCC expansions:")
for ncc in [rho_ref, T_ref, rho_ref*T_ref, Q]:
    ncc = ncc.evaluate()
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))


    
#########################################################
# 3D Anelastic hydrodynamics
problem = d3.IVP([p, S, u, L_buoy, L_diss, tau_p, tau_u1, tau_S1, tau_u2, tau_S2, tau_Lb, tau_Ld], namespace=locals())
	
problem.add_equation("dt(u) + grad(p) - RaPr*S*ez + T12*cross(0*ex + np.cos(Lat)*ey + np.sin(Lat)*ez,u) \
	- (1/rho_ref)*div(rho_ref*(grad(u) + trans(grad(u) ))  ) \
	+ twothirds*(1/rho_ref)*grad(rho_ref*div(u)) \
	  + lift(tau_u2,-2) + lift(tau_u1,-1)\
	= - u@grad(u) ")
		
			
problem.add_equation("Pr*rho_ref*T_ref*dt(S) - div(rho_ref*T_ref*grad(S))  + lift(tau_S2,-2)  + lift(tau_S1,-1) = Q \
			- Pr*rho_ref*T_ref*u@grad(S)  \
			+ Pr2thRa*( \
				trace(      grad(u)@(   ( grad(u) + trans(grad(u)) )*rho_ref    )      )  \
				- twothirds*rho_ref*(div(u)**2) \
				) \
			")
			
problem.add_equation("dot(grad(rho_ref), u) + rho_ref*trace(grad(u)) + lift(tau_u2,-1)@ez + tau_p = 0")


problem.add_equation("  dz(L_buoy) + lift(tau_Lb,-1) = - S*rho_ref*(u@ez) ")

problem.add_equation(" dz(L_diss) + lift(tau_Ld,-1) = \
	-trace(grad(u)@(grad(u) + trans(grad(u))) ) - twothirds*rho_ref*(div(u)**2)")


#########################################################
#boundary conditions
#problem.add_equation("S(z=Lz) = 0")
#problem.add_equation("dz(S)(z=0) = -1")
problem.add_equation("S(z=Lz) = 0")
problem.add_equation("dz(S)(z=0) = 0")

problem.add_equation("(u@ez)(z=0) = 0")
problem.add_equation("(u@ez)(z=Lz) = 0")
problem.add_equation("dz(u@ex)(z=0) = 0")
problem.add_equation("dz(u@ex)(z=1) = 0")
problem.add_equation("dz(u@ey)(z=0) = 0")
problem.add_equation("dz(u@ey)(z=1) = 0")
problem.add_equation("L_buoy(z=0) = 0")       # BC for L_buoy for partial depth integration
problem.add_equation("L_diss(z=0) = 0")       # BC for L_diss for partial depth integration
problem.add_equation("integ(p) = 0")  


#########################################################
solver = problem.build_solver(d3.RK222, ncc_cutoff=ncc_cutoff)
logger.info('Solver built')


#########################################################
if fresh_simulation:
    ##############################
    # this version produces random wavemodes
    # filtered to some lengthscale tapered
    # in z by a Gaussian
    ##############################
    kx=0*np.array(range(Nx))
    for i in range(Nx):
         kx[i]=np.mod((i) + Nx/2,Nx) - Nx/2;
         
    ky=0*np.array(range(Ny))
    for i in range(Ny):
         ky[i]=np.mod((i) + Ny/2,Ny) - Ny/2;         
    
    
    rand = np.random.RandomState(seed=41)
    #rand = np.random.RandomState(seed=int(time.time()))  # use this to init by clock
    noise_amp = rand.standard_normal( size=(Nx, Ny))
    #noise = rand.uniform( size=(Nx, Ny),  low=-1, high=1 ) # a different generator for random array
    
    noise_amp = noise_amp/np.amax(abs(noise_amp)) # normalises to maximum amplitude of 1
    phase = rand.uniform(high = 2*np.pi, size=(Nx, Ny))
    
    noise = noise_amp*np.exp(1j*phase)   
    
    for i in range(Nx):
        for j in range(Ny):  
            if kx[i] == 0 and ky[j] == 0:
                noise[i,j] = 0
            elif np.sqrt(kx[i]**2 + ky[j]**2) < rpf.modefilter:
                noise[i,j] = noise[i,j]
            else:
                noise[i,j] = 0
                
    noise = np.real(np.fft.ifft2(noise))
    noise = noise/np.max(noise)
    
    
    # full global coordinate arrays matching how the noise was built
    noise_x = np.linspace(0, Lx, Nx+1)[:-1]
    noise_y = np.linspace(0, Ly, Ny+1)[:-1]

    # local grid coordinates this rank owns (D3: x, y from dist.local_grids)
    local_x = x
    local_y = y

    tolerance_x = noise_x[1]/10
    tolerance_y = noise_y[1]/10

    for i in range(S['g'].shape[0]):
        current_x = local_x[i, 0, 0]
        index_x = np.where(np.abs(noise_x - current_x) <= tolerance_x)[0][0]
        for j in range(S['g'].shape[1]):
            current_y = local_y[0, j, 0]
            index_y = np.where(np.abs(noise_y - current_y) <= tolerance_y)[0][0]
            for k in range(S['g'].shape[2]):
                S['g'][i,j,k] = noise[index_x,index_y]

    # taper the z dependence with a gaussian. We also apply a top hat filter at the upper and lower boundary
    # to enforce 0 at these limits
    S['g'] = S['g']*rpf.pertamp_noise*np.exp(-100.0*(z-0.5)*(z-0.5))*np.piecewise(z,[ (z>=0.9) | (z<0.1) ,(z<0.9) & (z>=0.1)],[0,1])  
    
    S_bar = 0.0
    S['g'] = S['g'] + S_bar

    ##############################
    # end of initial condition setup
    ##############################


    # Initial timestep
    dt = rpf.initial_timestep

    # Integration parameters --- Note if these are all set to np.inf, simulation will perpetually run.
    solver.stop_sim_time = rpf.end_sim_time
    solver.stop_wall_time = (rpf.end_wall_time-1)*60*60 + 55*60
    solver.stop_iteration = rpf.end_iterations
    file_handler_mode = 'overwrite'

else:
    write, dt = solver.load_state(restart_file)
    solver.stop_sim_time = rpf.end_sim_time
    solver.stop_wall_time = (rpf.end_wall_time-1)*60*60 + 55*60
    solver.stop_iteration = rpf.end_iterations
    file_handler_mode = 'append'



#########################################################
# CFL criterion
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=rpf.safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=rpf.max_dt, threshold=0.05)
CFL.add_velocity(u)


#########################################################
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u), name='Re')

if debug_tau_terms:
    flow.add_property(tau_p, name='tau_p_flow')
    flow.add_property(tau_u1, name='tau_u1_flow')
    flow.add_property(tau_S1, name='tau_S1_flow')
    flow.add_property(tau_u2, name='tau_u2_flow')
    flow.add_property(tau_S2, name='tau_S2_flow')

#########################################################
# Saving checkpoints
checkpoints = solver.evaluator.add_file_handler(save_direc + 'checkpoints', wall_dt=rpf.checkpoint_freq, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

#########################################################
# Saving snapshots
snapshots = solver.evaluator.add_file_handler(save_direc + 'snapshots', sim_dt=rpf.snapshot_freq, max_writes=50, mode=file_handler_mode)
snapshots.add_task(u@ex, name='u')
snapshots.add_task(u@ey, name='v')
snapshots.add_task(u@ez, name='w')
snapshots.add_task(S, name='S')


#########################################################
# Saving timeseries
energies = solver.evaluator.add_file_handler(save_direc + 'energies', sim_dt=rpf.energies_freq, max_writes=5000, mode=file_handler_mode)
energies.add_task(hor_avg(S), name='mean_xy_s')
energies.add_task(vol_avg( np.sqrt(u@u) ), name='Re')
energies.add_task(0.5*vol_avg(rho_ref*(u@u)), name='KE')
energies.add_task(-Pr2thRa*vol_avg(trace(grad(u)@(( grad(u) + trans(grad(u)) )*rho_ref))  - twothirds*rho_ref*(div(u)**2) ),name='E_def')
energies.add_task(vol_avg(rho_ref*S*(u@ez))*Prth, name='E_F_conv')
#Magnitude of viscous dissipation as calculated by equation 5 (E_def) and equation 24 (E_F_conv) - See C&B '17

energies.add_task(np.sqrt(hor_avg(tau_u1@tau_u1)), name='tau_u1')
energies.add_task(np.sqrt(hor_avg(tau_u2@tau_u2)), name='tau_u2')
energies.add_task(np.sqrt(hor_avg(tau_S1**2)),     name='tau_S1')
energies.add_task(np.sqrt(hor_avg(tau_S2**2)),     name='tau_S2')
energies.add_task(np.sqrt(tau_p**2),               name='tau_p')



#########################################################
# Saving profiles
profiles = solver.evaluator.add_file_handler(save_direc + 'profiles', sim_dt=rpf.profiles_freq, max_writes=5000, mode=file_handler_mode)
profiles.add_task(hor_avg( u@ex), name='u_bar')
profiles.add_task(hor_avg( u@ey), name='v_bar')
profiles.add_task(-Pr2thRa*hor_avg(trace(grad(u)@(( grad(u) + trans(grad(u)) )*rho_ref))  - twothirds*rho_ref*(div(u)**2) ), name='E_def_z')

profiles.add_task(dz(hor_avg(S)), name='dz_Sbar')


#########################################################
# Saving fluxes
fluxes = solver.evaluator.add_file_handler(save_direc + 'fluxes', sim_dt=rpf.profiles_freq, max_writes=5000, mode=file_handler_mode)
fluxes.add_task(Pr*hor_avg(rho_ref*T_ref*S*(u@ez)), 	name='L_conv')
fluxes.add_task(hor_avg((-1)*rho_ref*T_ref*dz(S)), 	name='L_cond')
fluxes.add_task(hor_avg(L_buoy - L_buoy(z=0))*Pr2thRa, 		name='L_buoy')
fluxes.add_task(hor_avg(L_diss - L_diss(z=0))*Pr2thRa, 		name='L_diss')
fluxes.add_task(hor_avg(0.5*rho_ref*(u@u)*(u@ez))*Pr2thRa, 	name='L_KE')
fluxes.add_task(hor_avg(p*(u@ez))*Pr2thRa, 			name='L_p')                   
fluxes.add_task(hor_avg(rho_ref*(u@ez)*T_ref*S)*Pr + hor_avg(p*(u@ez))*Pr2thRa, name='L_enth')
fluxes.add_task((Pr2thRa/rho_ref)*hor_avg(twothirds*grad(rho_ref*div(u))-div(rho_ref*(grad(u) + trans(grad(u) ))  ) ),name='L_visc')

                   
#########################################################
# Saving slices
slices = solver.evaluator.add_file_handler(save_direc + 'slices', sim_dt=rpf.slices_freq, max_writes=5000, mode=file_handler_mode)
#slices = solver.evaluator.add_file_handler(save_direc + 'slices', iter=100, max_writes=5000, mode=file_handler_mode)
slices.add_task((u@ex)(y=0),  name="u_vert_xz")
slices.add_task((u@ey)(y=0),  name="v_vert_xz")
slices.add_task((u@ez)(y=0),  name="w_vert_xz")
slices.add_task((S)(y=0),     name="S_vert_xz")


slices.add_task((u@ex)(x=0),  name="u_vert_yz")
slices.add_task((u@ey)(x=0),  name="v_vert_yz")
slices.add_task((u@ez)(x=0),  name="w_vert_yz")
slices.add_task(S(x=0),       name="S_vert_yz")



slices.add_task((u@ex)(z=0.8),  name="u_horiz_xy_0p8")
slices.add_task((u@ey)(z=0.8),  name="v_horiz_xy_0p8")
slices.add_task((u@ez)(z=0.8),  name="w_horiz_xy_0p8")
slices.add_task(S(z=0.8),       name="S_horiz_xy_0p8")


slices.add_task((u@ex)(z=0.3),  name="u_horiz_xy_0p3")
slices.add_task((u@ey)(z=0.3),  name="v_horiz_xy_0p3")
slices.add_task((u@ez)(z=0.3),  name="w_horiz_xy_0p3")
slices.add_task(S(z=0.3),       name="S_horiz_xy_0p3")


slices.add_task((u@ex)(z=0.5),  name="u_horiz_xy_0p5")
slices.add_task((u@ey)(z=0.5),  name="v_horiz_xy_0p5")
slices.add_task((u@ez)(z=0.5),  name="w_horiz_xy_0p5")
slices.add_task(S(z=0.5),       name="S_horiz_xy_0p5")


slices.add_task((u@ex)(z=0.0),  name="u_horiz_xy_0p0")
slices.add_task((u@ey)(z=0.0),  name="v_horiz_xy_0p0")
slices.add_task((u@ez)(z=0.0),  name="w_horiz_xy_0p0")
slices.add_task(S(z=0.0),       name="S_horiz_xy_0p0")


slices.add_task((u@ex)(z=1.0),  name="u_horiz_xy_1p0")
slices.add_task((u@ey)(z=1.0),  name="v_horiz_xy_1p0")
slices.add_task((u@ez)(z=1.0),  name="w_horiz_xy_1p0")
slices.add_task(S(z=1.0),       name="S_horiz_xy_1p0")





#########################################################
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)

        if (solver.iteration) == 1:
            # Prints various parameters to terminal upon starting the simulation
            logger.info('Parameter values imported form run_param_file.py:')
            logger.info('Lx = {}, Ly={}, Lz = {}; (Resolution of {},{},{})'.format(Lx,Ly, Lz, Nx,Ny, Nz))
            logger.info('Ra = {}, Pr = {}, Np = {}'.format(Ra, Pr, Np))
            if rpf.end_sim_time != np.inf:
                logger.info('Simulation finishes at sim_time = {}'.format(rpf.end_sim_time))
            elif rpf.end_wall_time != np.inf:
                logger.info('Simulation finishes at wall_time = {}'.format(rpf.end_wall_time))
            elif rpf.end_iterations != np.inf:
                logger.info('Simulation finishes at iteration {}'.format(rpf.end_iterations))
            else:
                logger.info('No clear end point defined. Simulation may run perpetually.')

        if (solver.iteration-1) % 100 == 0:
            # Prints progress information include maximum Reynolds number every 10 iterations
            max_Re = flow.max('Re')
            logger.info('Iter: {:10d} Time: {:e}, dt: {:e}, Max Re = {:e}, Run time: {:f}'.format(solver.iteration, solver.sim_time, dt, flow.max('Re'), (time.time()-start_time)))
            
            if debug_tau_terms:
                tau_p, tau_u1, tau_S1, tau_u2, tau_S2
                maxtau_p = flow.max('tau_p_flow')
                maxtau_u1 = flow.max('tau_u1_flow')
                maxtau_S1 = flow.max('tau_S1_flow')
                maxtau_u2 = flow.max('tau_u2_flow')
                maxtau_S2 = flow.max('tau_S2_flow')
                #logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, dt, max_Re))
                logger.info('maxtau_p=%f, maxtau_u1=%f, maxtau_S1=%f, maxtau_u2=%f, maxtau_S2=%f, ' %(maxtau_p, maxtau_u1, maxtau_S1, maxtau_u2, maxtau_S2))
                logger.info('==========================================' )
            
            
            if np.isnan(flow.max('Re')):
                logger.info('NAN')
                raise Exception('Simulation has diverged. Re energy is NA')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    try:
        solver.log_stats()
    except AttributeError:
        logger.info("skipping log_stats (run shorter than warmup_iterations)")

















