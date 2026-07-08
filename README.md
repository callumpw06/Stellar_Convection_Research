# Stellar_Convection_Research
This is the GitHub repository containing documentation, code and material used to conduct a Research Project into the Adjoint-Based Maximisation of Heat Flux in Models of Stellar Convection. Throughout the project, I will document my background research, experiments, and conclusions on this repository. This document demonstrates steps taken throughout the project, including tutorials on how to run simulations on local devices - please the Examples folder.

## Weekly Progress Reports
Below, there is weekly progress reports, including; completed tasks, background reading and experiments.

### Week 1
29/06/26
- Read Chapter 1 of 'Internally heated convection and Rayleigh-Benard convection' by **David Goluskin**
- Worked through 'Tutorials' section of the [Dedalus Project website](https://dedalus-project.readthedocs.io/en/latest/pages/tutorials.html)
- Brief overview of [The Tau Method](https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html) used for imposing boundary conditions in PDE problems.

30/06/26
- Attempted to create Rayleigh-Benard simulations in Spherical and Polar Coordinates, without much success. Then, I attempted to use the Boussinesq equations to try model Rayleigh-Benard convection in 2D Cartesian coordinates; leading to diverging temperature values.
- Returned to the working Rayleigh-Benard code, and created working visualisations of buoyancy, vorticity, and velocity magnitude that can be found in the [Rayleigh-Benard](https://github.com/callumpw06/Stellar_Convection_Research/tree/main/Rayleigh-Benard) folder. This also includes a plot of Nusselt number against time for this particular simulation.

01/07/26
- Read through 'Fast automated adjoints for spectral PDE solvers' by **Calum S. Skene** and **Keaton J. Burns**
- Began reading 'Wall-to-wall optimal transport in two dimensions' by **Andre N. Souze**, **Ian Tobasco**, and **Charles R. Doering**

02/07/26
- Watched a [video](https://www.youtube.com/watch?v=Yiz92Ekn7vU) on adjoint-based optimisation for CFD examples, using an airfoil as an example.
- Research into why the temperature reading was diverging from my code using the Boussinesq equations - it was because of the different nondimensionalisation to the example script. Rescaling time (to much smaller time steps) has fixed this issue - there is visualisations of this code depicted in the [Boussinesq_Equations](https://github.com/callumpw06/Stellar_Convection_Research/tree/main/Boussinesq_Equations) folder.

03/07/26
- Continued reading the paper by **David Goluskin**.
- Re-read 'Fast automated adjoints for spectral PDE solvers' and used Google Gemini to research the adjoint methods for optimising an arbitrary objective function using SciPy and the dedalus3 python libraries.
- Began coding my own program to optimise the Nusselt Number in the Boussinesq equations simulations. It runs currently, but my laptop doesn't computational power to complete the program.

### Week 2
06/07/26
- Read through part of the paper 'Optimal mixing in two-dimensional stratified plane Poiseuille flow at finite Peclet and Richardson numbers' by **F. Marcotte** and **C. P. Caulfield**
- Obtained the working Optimal Mixing example to find the 'best' initial conditions for mixing fluids.

07/07/26
- Fixed errors in plotting the Nusselt number against time in the Boussinesq equations folder, so it now creates a reasonable plot.
- Created a rough plan for the structure of the paper I will write.

08/07/26
- Read through 'Stellar Convection' by **Eric Graham** to gain an understanding of the roll of convection currents in stars.
- 