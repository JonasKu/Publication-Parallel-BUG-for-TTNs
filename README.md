# Publication Parallel BUG for TTNs

This repository collects the numerical test cases for the manuscript "Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks". 

1. **Section 5.1**: The code uses Matlab. To run the examples one must add the folders "parallel_TTN_integrator" via the addpath(...) function from Matlab. Further, one must download and also add the tensor toolboxes [Tensorlab](https://www.tensorlab.net/#download), [Tensortoolbox](https://www.tensortoolbox.org/), and [hm-toolbox](https://github.com/numpi/hm-toolbox). Further, the folder "TTNO", which is another repository (find it [here](https://github.com/DominikSulz/TTNO)), has to be added. After including all folders, one can run the examples by running the scripts: run_script_long_range_Ising.m for the Ising model and runscript_error_plot.m for the error plot.
2. **Sections 5.2 and 5.3**: The numerical experiments are computed with Julia, Version 1.10.4. To run all experiments, simply go into one of the directories and type julia main.jl into the command line. This will install all required packages, run all computations, and reproduce all figures from the manuscript.
