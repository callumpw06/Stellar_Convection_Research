"""
This script will solve the Boussinesq equations for convection
in a 2D domain using the finite difference method.

This script can be run on MPI through the command:
    mpirun -np <number_of_processes> python Boussinesq_Convection.py <Lx>
"""

import sys
import numpy as np
import dedalus.public as d3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Parse Lx from the command line. Default to 2.0 if not provided.
if len(sys.argv) > 1:
    try:
        Lx = float(sys.argv[1])
    except ValueError:
        logger.error(f"Invalid Lx argument provided: {sys.argv[1]}. Must be a float.")
        sys.exit(1)
else:
    Lx = 2.0
    logger.info("No Lx argument provided. Defaulting to Lx = 2.0")

# Parameters
Lz = 1
# Scale Nx proportionally to Lx to maintain grid resolution
Nx, Nz = 128, 64
Rayleigh = 1e5
Prandtl = 1
Q = 0
dealias = 3/2
stop_sim_time = 1.0  # Time allowed for the fluid to settle into steady state
timestepper = d3.RK222
max_timestep = 1e-3
dtype = np.float64

logger.info(f"Running simulation with Lx = {Lx}, Nx = {Nx}, Nz = {Nz}")

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
T = dist.Field(name='T', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1) # First-order reduction

# Problem
problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - div(grad_T) + lift(tau_T2) = - u@grad(T) + Q")
problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*T*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("T(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("T(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T['g'] *= z * (Lz - z) # Damp noise at walls
T['g'] += Lz - z # Add linear background

# CFL
CFL = d3.CFL(solver, initial_dt=1e-7, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    
    # Calculate when to start recording (e.g., 5.0 - 0.5 = 4.5)
    analysis_setup = False
    analysis_start_time = max(0.0, stop_sim_time - 0.5)
    
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        # Start recording data only during the last 0.5 time units
        if not analysis_setup and solver.sim_time >= analysis_start_time:
            logger.info(f"Starting analysis output at t = {solver.sim_time:e}")
            out_dir = Path("analysis_runs") / f"Lx_{Lx}"
            analysis = solver.evaluator.add_file_handler(out_dir, sim_dt=max_timestep, max_writes=50)
            analysis.add_task(T, name='temperature')
            analysis.add_task(u@ez, name='w_velocity')
            analysis.add_task(-d3.div(d3.skew(u)), name='vorticity')
            analysis.add_task(np.sqrt(u@u), name='velocity_magnitude')
            analysis.add_task(1 + d3.integ(u@ez * T)/(Lx*Lz), name='Nu') # Nusselt number
            analysis.add_task(p, name='pressure')
            analysis.add_task(-d3.integ(d3.grad(T)@ez, 'x')(z=Lz) / Lx, name='heat_flux_top')
            analysis_setup = True
            
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info(f'Iteration={solver.iteration}, Time={solver.sim_time:e}, dt={timestep:e}, Completed={100*solver.sim_time/solver.stop_sim_time:.2f}%, max(Re)={max_Re:.2f}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()