# Nonlinear FEM Simulation of Power-Law Creep (1D Bar) — Python

This project implements a small, self-contained finite element (FEM) solver for a 1D two-segment bar subjected to a time-dependent point load, including **Norton / power-law creep** with implicit time integration and Newton–Raphson solution strategies. [file:15][file:14]

## Problem overview
A fixed–fixed bar consisting of two segments (different lengths/cross-sections) is loaded at the interface node by a force ramp up to a final value and then held constant until the end of the simulation. [file:15]  
The constitutive model decomposes total strain into elastic + creep parts and evolves creep strain via a power-law creep relation (implicit update). [file:15][file:14]

## Key features
- 2-node rod elements with linear shape functions. [file:15][file:14]
- Internal force vector and element stiffness via Gauss quadrature (1 or 2 Gauss points depending on the variant). [file:15][file:14]
- Power-law creep material routine with implicit Euler-Backward time integration and a **local Newton–Raphson** solver for the creep update. [file:15][file:14]
- Algorithmically consistent tangent stiffness for stable convergence of the global Newton–Raphson loop. [file:15][file:14]
- Verification (linear-elastic limit vs. analytical solution) and convergence studies (h- and time-step convergence). [file:15][file:14]

## Repository / code structure
The implementation is modular and follows the typical FEM separation:
- **Material routine**: updates stress, creep strain, and consistent tangent from strain and previous state. [file:14]
- **Element routine**: computes element internal force vector and stiffness matrix using the material routine and Gauss integration. [file:14]
- **Main driver**: time stepping loop, global assembly, boundary conditions, Newton–Raphson iterations, and plotting/output. [file:14]

In the documented setup, the work is organized into separate scripts for verification and studies:
- `linear_elastic.py` — baseline verification with creep disabled. [file:14]
- `creep_non_linear.py` — nonlinear creep simulation (primary model). [file:14]
- `convergence.py` — mesh/time-step convergence routines. [file:14]

(If file names differ in your repo, rename these entries accordingly.)

## Requirements
- Python 3.x [file:14]
- NumPy [file:14]
- Matplotlib [file:14]

## How to run
From a terminal in the project folder:
```bash
python creep_non_linear.py
python linear_elastic.py
python convergence.py

