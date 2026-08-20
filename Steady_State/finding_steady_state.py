import numpy as np
import dedalus.public as d3
import logging
import sys

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Plotting Configuration ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})

# --- HPC Safe Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Problem Parameters ---
Lx = 3.0        
Lz = 1.0        
Nx = 128         
Nz = 96         
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

# --- Continuation Fields ---
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

# =====================================================================
# --- Boundary Conditions (Free-Slip) ---
# Free-slip requires impermeability (w = 0) and zero shear stress (du/dz = 0)
u_z = ez @ (grad_v @ ex)
w = ez @ v

problem.add_equation("w(z=0) = 0")
problem.add_equation("u_z(z=0) = 0")
problem.add_equation("w(z=Lz) = 0")
problem.add_equation("u_z(z=Lz) = 0")
problem.add_equation("T(z=0) = 1")
problem.add_equation("T(z=Lz) = 0")
problem.add_equation("integ(p) = 0")
# =====================================================================

# --- Initial Seed ---
A_roll = 5.0  
T['g'] = 1.0 - z_norm + A_roll * np.cos(k_x * x) * np.sin(np.pi * z_norm)
v['g'][0] = -A_roll * np.sin(k_x * x) * (2 * z_norm * (1 - z_norm)**2 - 2 * (z_norm**2) * (1 - z_norm)) / Lz
v['g'][1] =  A_roll * k_x * np.cos(k_x * x) * (z_norm**2) * ((1 - z_norm)**2)
p['g'] = 0.0

# Build the solver once.
solver = problem.build_solver(ncc_cutoff=1e-3)

# =====================================================================
# --- The "Defect" Continuation Schedule (High Density) ---
schedule = [
    (3000.0, 0.5),   
    (3000.0, 0.0),   
    (5000.0, 0.0),   
    (7500.0, 0.0),  
    (10000.0, 0.0)
]

# Generate 45 logarithmically spaced points between 12,000 and 10,000,000
high_density_Ra = np.logspace(np.log10(12000.0), np.log10(1e7), num=20)
for target in high_density_Ra:
    schedule.append((target, 0.0))
# =====================================================================

tolerance = 1e-7
max_steps = 10

Nu_integrand = d3.integ(1 + (v@ez) * T) / (Lx * Lz)

recorded_Ra = []
recorded_Nu = []

for target_Ra, target_eps in schedule:
    Ra['g'] = target_Ra
    epsilon['g'] = target_eps
    
    logger.info(f"=== NLBVP Continuation: Ra = {target_Ra:.1e} | Forcing = {target_eps:.1f} ===")
    
    for step in range(1, max_steps + 1):
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        
        logger.info(f"  Step {step:02d}/{max_steps}: Perturbation norm = {pert_norm:.3e}")
        
        if pert_norm <= tolerance:
            logger.info(f"  -> Converged perfectly in {step} steps!")
            break
    else:
        logger.error(f"Solver failed to converge at Ra={target_Ra} after {max_steps} steps.")
        sys.exit(1)
        
    if target_eps == 0.0:
        Nu_val_array = Nu_integrand.evaluate()['g']
        local_Nu = Nu_val_array.flatten()[0] if Nu_val_array.size > 0 else 0.0
        global_Nu = dist.comm.bcast(local_Nu, root=0)
        
        if dist.comm.rank == 0:
            recorded_Ra.append(target_Ra)
            recorded_Nu.append(global_Nu)
            logger.info(f"  Recorded Nu = {global_Nu:.4f} for Ra = {target_Ra:.1e}")

logger.info("Convective steady state sweep successfully finished!")

if dist.comm.rank == 0:
    logger.info("Saving Ra and Nu data to 'Nu_vs_Ra_data.csv'...")
    export_data = np.column_stack((recorded_Ra, recorded_Nu))
    np.savetxt('Nu_vs_Ra_data.csv', export_data, header='Rayleigh,Nusselt', delimiter=',', comments='')

# --- 1. Plotting: Nusselt vs Rayleigh ---
if dist.comm.rank == 0:
    logger.info("Generating Nu vs Ra plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(recorded_Ra, recorded_Nu, marker='o', linestyle='-', color='#d62728', linewidth=2, markersize=5)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Rayleigh Number ($Ra$)')
    plt.ylabel('Nusselt Number ($Nu$)')
    plt.title('Nusselt Number vs. Rayleigh Number (Free-Slip)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Nu_vs_Ra_sweep.png', dpi=300)
    logger.info("Saved Nu_vs_Ra_sweep.png")

# --- 2. Visualization: Final Fluid State (MPI Safe) ---
T.change_scales(1)
v.change_scales(1)

T_data = T.allgather_data('g')
vx_data = v.allgather_data('g')[0]
vz_data = v.allgather_data('g')[1]

if dist.comm.rank == 0:
    logger.info("Generating final fluid state visualization...")
    import scipy.interpolate as interp
    
    x_global = xbasis.global_grid(dist, scale=1).flatten()
    z_global = zbasis.global_grid(dist, scale=1).flatten()
    
    x_plot = np.append(x_global, Lx)
    T_plot = np.vstack([T_data, T_data[0:1, :]])
    vx_plot = np.vstack([vx_data, vx_data[0:1, :]])
    vz_plot = np.vstack([vz_data, vz_data[0:1, :]])
    speed_plot = np.sqrt(vx_plot**2 + vz_plot**2)
    
    X, Z = np.meshgrid(x_plot, z_global, indexing='ij')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    mesh1 = ax1.pcolormesh(X, Z, T_plot, shading='gouraud', cmap='RdBu_r')
    fig.colorbar(mesh1, ax=ax1, label='Temperature ($T$)')
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Lz)
    ax1.set_ylabel('Height ($z$)')
    ax1.set_title(f'Final Steady-State Temperature ($Ra = {target_Ra:.1e}, Pr = {Pr}$)')

    mesh2 = ax2.pcolormesh(X, Z, speed_plot, shading='gouraud', cmap='viridis')
    fig.colorbar(mesh2, ax=ax2, label='Speed ($|v|$)')

    z_uniform = np.linspace(0, Lz, len(z_global))
    sort_idx = np.argsort(z_global)
    z_sorted = z_global[sort_idx]
    
    vx_sorted = vx_plot[:, sort_idx]
    vz_sorted = vz_plot[:, sort_idx]
    
    f_vx = interp.interp1d(z_sorted, vx_sorted, axis=1, kind='cubic', fill_value='extrapolate')
    f_vz = interp.interp1d(z_sorted, vz_sorted, axis=1, kind='cubic', fill_value='extrapolate')
    
    vx_uni = f_vx(z_uniform)
    vz_uni = f_vz(z_uniform)
    speed_uni = np.sqrt(vx_uni**2 + vz_uni**2)
    
    ax2.streamplot(x_plot, z_uniform, 
                   vx_uni.T, vz_uni.T, color='white', density=1.2, 
                   linewidth=1.5 * speed_uni.T / speed_uni.max())

    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Lz)
    ax2.set_xlabel('Position ($x$)')
    ax2.set_ylabel('Height ($z$)')
    ax2.set_title('Velocity Field & Streamlines (Free-Slip)')

    plt.tight_layout()
    output_filename = 'final_steady_state_convection.png'
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Saved {output_filename}")