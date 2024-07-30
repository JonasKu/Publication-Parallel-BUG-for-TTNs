# Publication Parallel BUG for TTNs

This repository collects the numerical test cases for the manuscript "Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks". 

Section 5.1: The code uses Matlab. Navigate into the folder parallel_TTN_integrator_test_cases/Ising_long_range and run run_script_long_range_Ising.m and runscript_error_plot.m in Matlab.

Sections 5.2 and 5.3: The numerical experiments are computed with Julia, Version 1.10.4. To run all experiments, simply go into one of the directories and type julia main.jl into the command line. This will install all required packages, run all computations, and reproduce all figures from the manuscript.
