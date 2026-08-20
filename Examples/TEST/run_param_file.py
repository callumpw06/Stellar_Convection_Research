#Dedalus3
"""
Parameter file for use in the Dedalus 2D anelastic convection script.
"""

import numpy as np

Lx, Ly, Lz = 1.00, 1.00, 1                       # Domain size
Nx, Ny, Nz = 32, 32, 32                   # Number of

"""
If you want to run on a different number of cores, 
then you will need to change meshx and meshy in 
run_param_file.py so that their product is equal
to the number of cores. Currently as set up the 
run_param_file.py needs to be in the same directory as the main script. 
"""
meshx =    1
meshy =    8

Pr = 1.000000000                             # Prandtl number
Ra = 10000.000000000                          # Rayleigh number
Np = 1.000000000                             # Number of density scale heights
m = 1.500000000                             # Polytropic index
theta = 1 - np.exp(-Np/m)           # Dimensionaless inverse T scale height

Q_amp = 1

Ta = 100.000000000
latitude = 1.570000000

initial_timestep = 1.5e-5                 # Initial timestep
max_dt = 1e-3                          # max dt

checkpoint_freq = 3600.000000000              # Frequency snapshot files are outputted
snapshot_freq = 1.000000000              # Frequency snapshot files are outputted
energies_freq = 0.010000000              # Frequency analysis files are outputted
profiles_freq = 0.010000000             # Frequency analysis files are outputted
fluxes_freq = 0.010000000              # Frequency analysis files are outputted
slices_freq = 0.010000000              # Frequency analysis files are outputted

end_sim_time = 1.000000000                   # Stop time in simulations units
end_wall_time = 1.00000              # Stop time in wall time
end_iterations = np.inf              # Stop time in iterations (np.inf normally)

safety_factor = 0.200

fresh_simulation = True
restart_file = 'checkpoints_s0'

pertamp_vert = 0.000000000
pertamp_noise = 0.500000000
modefilter = 10

Dversion = 'D3'