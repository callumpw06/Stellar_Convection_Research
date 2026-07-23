"""
This script will solve the Boussinesq equations for convection
in a 2D domain using the finite difference method. The Boussinesq
equations describe the behavior of a fluid under the influence of
buoyancy forces, which are important in many geophysical and 
engineering applications.

We will use a simple explicit time-stepping scheme to evolve the 
system in time. The equations will be discretized on a uniform grid,
and we will apply appropriate boundary conditions.

The Boussinesq equations consist of the Navier-Stokes equations for
incompressible flow, coupled with a temperature equation that accounts 
for buoyancy effects. The primary variables are the velocity field 
(u), pressure (p), and temperature (T).

This script can be run on MPI through the command:
    mpirun -np <number_of_processes> python Boussinesq_Convectiom.py
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = 4, 1
Nx, Nz = 256, 64
Rayleigh = 3e5
Prandtl = 1
Q = 0
dealias = 3/2
stop_sim_time = 5.0  # Time allowed for the fluid to settle into steady state
timestepper = d3.RK222
max_timestep = 1e-5
dtype = np.float64

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
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
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

# Analysis
analysis = solver.evaluator.add_file_handler('analysis', sim_dt=10*max_timestep, max_writes=50)
analysis.add_task(T, name='temperature')
analysis.add_task(-d3.div(d3.skew(u)), name='vorticity')
analysis.add_task(np.sqrt(u@u), name='velocity_magnitude')
analysis.add_task(1 + d3.integ(u@ez * T)/(Lx*Lz), name='Nu') # Nusselt number
analysis.add_task(p, name='pressure')
analysis.add_task(-d3.integ(d3.grad(T)@ez, 'x')(z=Lz) / Lx, name='heat_flux_top')

# CFL
CFL = d3.CFL(solver, initial_dt=1e-7, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info(f'Iteration={solver.iteration}, Time={solver.sim_time:e}, dt={timestep:e}, Completed={100*solver.sim_time/solver.stop_sim_time:.2f}%, max(Re)={max_Re:.2f}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
