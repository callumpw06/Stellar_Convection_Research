# Stellar_Convection_Research

In this repository is attached the final drafts of the paper outlining the findings of this research project, including an Extended and a shorter, Report Version of the paper. 

This is the GitHub repository containing documentation, code and material used to conduct a Research Project into the Adjoint-Based Maximisation of Heat Flux in Models of Stellar Convection. Throughout the project, I will document my background research, experiments, and conclusions on this repository. This document demonstrates steps taken throughout the project, including tutorials on how to run simulations on local devices - please see Examples folder.

Instructions to run your own Rayleigh-Benard simulation:
    -> \Boussinesq_Equations
    `python3 Boussinesq_Convection.py <L>`
One can change Rayleigh number, Taylor number and resolution within the script itself.
Includes other python scripts to visualise temperature, pressure, velocity, vorticity, nusselt number, etc.

Instructions to run your own Adjoint-Based optimisation:
    -> \Adjoint_Problems
    Domain width optimisation:
        `python3 optimise_L.py`
One can change L_0, max_iterations within the script.
One can change resolution Rayleigh, Prandtl, Taylor numbers in the `forward_solver.py <L> <restart> <BC>` script (as well as internal heating).
    
    Initial conditions optimisation:
        -> \Optimising_Initial_Conditions
        `python3 wrapper.py`
One can change boundary conditions, etc. within the script itself.