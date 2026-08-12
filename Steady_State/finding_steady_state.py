import numpy as np
import dedalus.public as d3
import logging
import sys

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- HPC Safe Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Problem Parameters ---
Lx = 3.0        
Lz = 1.0        
Nx = 64         
Nz = 64         
Pr = 1.0        
dealias = 3/2
dtype = np.float64

# --- Coordinates & Domain Bases ---
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# --- Fields Definition ---
p = dist.Field(name='p', bases=(xbasis, zbasis))
v = dist.VectorField(coords, name='v', bases=(xbasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, zbasis))

grad_v = dist.TensorField(coords, name='grad_v', bases=(xbasis, zbasis))
grad_T = dist.VectorField(coords, name='grad_T', bases=(xbasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_v1 = dist.VectorField(coords, name='tau_v1', bases=xbasis)
tau_v2 = dist.VectorField(coords, name='tau_v2', bases=xbasis)
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

# --- Continuation Fields (Correct Dedalus 3 Method) ---
Ra = dist.Field(name='Ra')
epsilon = dist.Field(name='epsilon')

# --- Artificial Forcing Field (The "Thermal Magnet") ---
x, z = dist.local_grids(xbasis, zbasis)
k_x = 2.0 * np.pi / Lx
z_norm = z / Lz

forcing_T = dist.Field(name='forcing_T', bases=(xbasis, zbasis))
forcing_T['g'] = np.cos(k_x * x) * np.sin(np.pi * z_norm) 

# --- Operators & Substitutions ---
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# --- Problem Setup ---
namespace = locals()
problem = d3.NLBVP([p, v, T, grad_v, grad_T, tau_p, tau_v1, tau_v2, tau_T1, tau_T2], namespace=namespace)

problem.add_equation("trace(grad_v) + tau_p = 0")
problem.add_equation("grad_v - grad(v) - ez * lift(tau_v1) = 0")
problem.add_equation("v @ grad(v) + grad(p) - Pr * div(grad_v) - Pr * Ra * T * ez + lift(tau_v2) = 0")
problem.add_equation("grad_T - grad(T) - ez * lift(tau_T1) = 0")
problem.add_equation("v @ grad(T) - div(grad_T) - epsilon * forcing_T + lift(tau_T2) = 0")

# --- Boundary Conditions (No-Slip) ---
problem.add_equation("v(z=0) = 0")
problem.add_equation("v(z=Lz) = 0")
problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

# --- Initial Seed ---
A_roll = 5.0  
T['g'] = 1.0 - z_norm + A_roll * np.cos(k_x * x) * np.sin(np.pi * z_norm)
v['g'][0] = -A_roll * np.sin(k_x * x) * (2 * z_norm * (1 - z_norm)**2 - 2 * (z_norm**2) * (1 - z_norm)) / Lz
v['g'][1] =  A_roll * k_x * np.cos(k_x * x) * (z_norm**2) * ((1 - z_norm)**2)
p['g'] = 0.0

# Build the solver once. It will dynamically re-evaluate the fields in the loop!
solver = problem.build_solver(ncc_cutoff=1e-3)

# --- The "Defect" Continuation Schedule ---
schedule = [
    (3000.0, 0.5),   # Phase 1: Force the rolls to form and lock their position
    (3000.0, 0.0),   # Phase 2: Remove the artificial forcing (snaps to true physics)
    (10000.0, 0.0),  # Phase 3: Push Ra up...
    (30000.0, 0.0),  # Phase 4: Intermediate step for safety
    (50000.0, 0.0),
    (100000.0, 0.0)  # Target state!
]

tolerance = 1e-8
max_steps = 20

for target_Ra, target_eps in schedule:
    # Update the parameter fields dynamically
    Ra['g'] = target_Ra
    epsilon['g'] = target_eps
    
    logger.info(f"=== NLBVP Continuation: Ra = {target_Ra:.1e} | Forcing = {target_eps:.1f} ===")
    
    # Use a hard 'for' loop to guarantee it never exceeds max_steps
    for step in range(1, max_steps + 1):
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        
        logger.info(f"  Step {step:02d}/{max_steps}: Perturbation norm = {pert_norm:.3e}")
        
        # Explicit break condition
        if pert_norm <= tolerance:
            logger.info(f"  -> Converged perfectly in {step} steps!")
            break
    else:
        # The 'else' block on a for-loop only triggers if the loop NEVER hit a 'break'
        logger.error(f"Solver failed to converge at Ra={target_Ra} after {max_steps} steps.")
        sys.exit(1)

logger.info("Convective steady state successfully solved!")

# --- Visualization (MPI Safe) ---
T.change_scales(1)
v.change_scales(1)

T_data = T.allgather_data('g')
vx_data = v.allgather_data('g')[0]
vz_data = v.allgather_data('g')[1]

if dist.comm.rank == 0:
    logger.info("Generating visualization...")
    
    x_global = xbasis.global_grid(dist, scale=1).flatten()
    z_global = zbasis.global_grid(dist, scale=1).flatten()
    X, Z = np.meshgrid(x_global, z_global, indexing='ij')

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(X, Z, T_data, shading='gouraud', cmap='RdBu_r')
    plt.colorbar(label='Temperature (T)')

    speed = np.sqrt(vx_data**2 + vz_data**2)
    plt.streamplot(x_global, z_global, 
                   vx_data.T, vz_data.T, color='black', density=1.2, linewidth=1.5 * speed.T / speed.max())

    plt.xlabel('Position ($x$)')
    plt.ylabel('Height ($z$)')
    plt.title(f'NLBVP Steady-State Convection ($Ra = 10^5, Pr = {Pr}$)')
    plt.tight_layout()

    output_filename = 'steady_state_convection.png'
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Saved {output_filename}")