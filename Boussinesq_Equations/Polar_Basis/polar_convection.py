"""
This script will solve the Boussinesq equations for convection
in a 2D Polar (annular) domain using the finite difference method.

This script can be run on MPI through the command:
    mpirun -np <number_of_processes> python Polar_Boussinesq_Convection.py <Ro>
"""

import sys
import numpy as np
import dedalus.public as d3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default geometric parameters for the annulus
Ri = 0.01  # Inner radius
Ro = 1.0  # Outer radius

if len(sys.argv) > 1:
    try:
        Ro = float(sys.argv[1])
    except ValueError:
        logger.error(f"Invalid Ro argument provided: {sys.argv[1]}. Must be a float.")
        sys.exit(1)

L = Ro - Ri  # Gap width

# Parameters
Nphi, Nr = 128, 64
Rayleigh = 1e6
Prandtl = 1
Taylor = 1e5
Q = 0
dealias = 3/2
stop_sim_time = 1.0
analysis_duration = 1.0
timestepper = d3.RK222
max_timestep = 1e-3
dtype = np.float64

logger.info(f"Running simulation with Ro = {Ro}, Nphi = {Nphi}, Nr = {Nr}")

# -------------------------------------------------------------------------
# We use a Cartesian topology ('phi', 'r') to circumvent the origin-singularity 
# constraints of Dedalus 3's native PolarCoordinates. We will explicitly 
# inject the 1/r metric factors into the equations below.
# -------------------------------------------------------------------------
coords = d3.CartesianCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)

phi_basis = d3.RealFourier(coords['phi'], size=Nphi, bounds=(0, 2*np.pi), dealias=dealias)
r_basis = d3.ChebyshevT(coords['r'], size=Nr, bounds=(Ri, Ro), dealias=dealias)

# Scalar Fields for Radial and Azimuthal components
p = dist.Field(name='p', bases=(phi_basis, r_basis))
T = dist.Field(name='T', bases=(phi_basis, r_basis))
ur = dist.Field(name='ur', bases=(phi_basis, r_basis))
uphi = dist.Field(name='uphi', bases=(phi_basis, r_basis))

# Tau fields (only required for the Chebyshev/bounded dimension)
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=phi_basis)
tau_T2 = dist.Field(name='tau_T2', bases=phi_basis)
tau_ur1 = dist.Field(name='tau_ur1', bases=phi_basis)
tau_ur2 = dist.Field(name='tau_ur2', bases=phi_basis)
tau_uphi1 = dist.Field(name='tau_uphi1', bases=phi_basis)
tau_uphi2 = dist.Field(name='tau_uphi2', bases=phi_basis)

# Extract radial coordinate grid as a dynamic Dedalus Field for the 1/r terms
phi_grid, r_grid = dist.local_grids(phi_basis, r_basis)
r = dist.Field(name='r', bases=r_basis)
r['g'] = r_grid

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
Coriolis_coeff = Prandtl * np.sqrt(Taylor)

# Derivative operators
dphi = lambda A: d3.Differentiate(A, coords['phi'])
dphi2 = lambda A: d3.Differentiate(d3.Differentiate(A, coords['phi']), coords['phi'])
dr = lambda A: d3.Differentiate(A, coords['r'])

# First-order reductions with Tau polynomial lifts
lift_basis = r_basis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

dr_T = dr(T) + lift(tau_T1)
dr_ur = dr(ur) + lift(tau_ur1)
dr_uphi = dr(uphi) + lift(tau_uphi1)

# Manually construct Polar Laplacians using the Cartesian derivatives
lap_T = dr(dr_T) + dr_T/r + dphi2(T)/(r**2)
lap_ur = dr(dr_ur) + dr_ur/r + dphi2(ur)/(r**2) - ur/(r**2) - 2*dphi(uphi)/(r**2)
lap_uphi = dr(dr_uphi) + dr_uphi/r + dphi2(uphi)/(r**2) - uphi/(r**2) + 2*dphi(ur)/(r**2)

# Non-linear Advection terms
adv_T = ur*dr_T + (uphi/r)*dphi(T)
adv_ur = ur*dr_ur + (uphi/r)*dphi(ur) - (uphi**2)/r
adv_uphi = ur*dr_uphi + (uphi/r)*dphi(uphi) + (ur*uphi)/r

# Problem Setup
problem = d3.IVP([p, T, ur, uphi, tau_p, tau_T1, tau_T2, tau_ur1, tau_ur2, tau_uphi1, tau_uphi2], namespace=locals())

# Continuity Equation (Incompressible)
problem.add_equation("dr_ur + ur/r + dphi(uphi)/r + tau_p = 0")

# Heat Equation
problem.add_equation("dt(T) - lap_T + lift(tau_T2) = -adv_T + Q")

# Radial (ur) Momentum Equation
problem.add_equation("dt(ur) - Prandtl*lap_ur + dr(p) - Prandtl*Rayleigh*T - Coriolis_coeff*uphi + lift(tau_ur2) = -adv_ur")

# Azimuthal (uphi) Momentum Equation
problem.add_equation("dt(uphi) - Prandtl*lap_uphi + dphi(p)/r + Coriolis_coeff*ur + lift(tau_uphi2) = -adv_uphi")

# Boundary Conditions
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("ur(r=Ri) = 0")
problem.add_equation("ur(r=Ro) = 0")
problem.add_equation("uphi(r=Ri) = 0")
problem.add_equation("uphi(r=Ro) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T.fill_random('g', seed=42, distribution='normal', scale=1e-3)
T['g'] *= (r_grid - Ri) * (Ro - r_grid)
T['g'] += (Ro - r_grid) / L

# CFL (combine scalars back into a single vector expression for the cadence checker)
ephi, er = coords.unit_vector_fields(dist)
CFL = d3.CFL(solver, initial_dt=1e-7, cadence=10, safety=0.2, threshold=0.05, max_change=1.2, min_change=0.1, max_dt=max_timestep)
CFL.add_velocity(uphi*ephi + ur*er)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(ur**2 + uphi**2)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    
    analysis_setup = False
    analysis_start_time = max(0.0, stop_sim_time - analysis_duration)
    
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        # Start recording data only during the last 0.5 time units
        if not analysis_setup and solver.sim_time >= analysis_start_time:
            logger.info(f"Starting analysis output at t = {solver.sim_time:e}")
            out_dir = Path("analysis_runs") / f"Ro_{Ro}"
            analysis = solver.evaluator.add_file_handler(out_dir, sim_dt=max_timestep, max_writes=50)
            
            analysis.add_task(T, name='temperature')
            analysis.add_task(ur, name='u_radial')
            analysis.add_task(uphi, name='u_azimuthal')
            
            # 2D Vorticity: curl_z = d(u_phi)/dr + u_phi/r - (1/r)*d(u_r)/dphi
            vorticity = dr_uphi + uphi/r - dphi(ur)/r
            analysis.add_task(vorticity, name='vorticity')
            analysis.add_task(np.sqrt(ur**2 + uphi**2), name='velocity_magnitude')
            
            # Nusselt number over the true polar area element (r dr dphi)
            annulus_area = np.pi * (Ro**2 - Ri**2)
            analysis.add_task(1 + d3.integ(ur * T * r) / annulus_area, name='Nu') 
            analysis.add_task(p, name='pressure')
            
            # Outer boundary heat flux
            analysis.add_task(-d3.integ(dr_T, 'phi')(r=Ro) / (2 * np.pi), name='heat_flux_outer')
            analysis_setup = True
            
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info(f'Iteration={solver.iteration}, Time={solver.sim_time:e}, dt={timestep:e}, Completed={100*solver.sim_time/solver.stop_sim_time:.2f}%, max(Re)={max_Re:.2f}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()