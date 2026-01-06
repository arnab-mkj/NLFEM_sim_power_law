# Nonlinear FEM Simulation of Power-Law Creep (1D Bar) — Python

This project implements a small, self-contained finite element (FEM) solver for a 1D two-segment bar subjected to a time-dependent point load, including **Norton / power-law creep** with implicit time integration and Newton–Raphson solution strategies.

## Problem overview
A fixed–fixed bar consisting of two segments (different lengths/cross-sections) is loaded at the interface node by a force ramp up to a final value and then held constant until the end of the simulation.
The constitutive model decomposes total strain into elastic + creep parts and evolves creep strain via a power-law creep relation (implicit update). 

## Key features
- 2-node rod elements with linear shape functions. 
- Internal force vector and element stiffness via Gauss quadrature (1 or 2 Gauss points depending on the variant). 
- Power-law creep material routine with implicit Euler-Backward time integration and a **local Newton–Raphson** solver for the creep update.
- Algorithmically consistent tangent stiffness for stable convergence of the global Newton–Raphson loop. 
- Verification (linear-elastic limit vs. analytical solution) and convergence studies (h- and time-step convergence).
## Repository / code structure
The implementation is modular and follows the typical FEM separation:
- **Material routine**: updates stress, creep strain, and consistent tangent from strain and previous state. 
- **Element routine**: computes element internal force vector and stiffness matrix using the material routine and Gauss integration. 
- **Main driver**: time stepping loop, global assembly, boundary conditions, Newton–Raphson iterations, and plotting/output. 

In the documented setup, the work is organized into separate scripts for verification and studies:
- `linear_elastic.py` — baseline verification with creep disabled. 
- `creep_non_linear.py` — nonlinear creep simulation (primary model).
- `convergence.py` — mesh/time-step convergence routines.

(If file names differ in your repo, rename these entries accordingly.)


## How to run
From a terminal in the project folder:
```bash
python creep_non_linear.py
python linear_elastic.py
python convergence.py

