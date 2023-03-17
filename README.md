# TGV_QSM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://korbinian90.github.io/TGV_QSM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://korbinian90.github.io/TGV_QSM.jl/dev/)
[![Build Status](https://github.com/korbinian90/TGV_QSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/korbinian90/TGV_QSM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/korbinian90/TGV_QSM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/korbinian90/TGV_QSM.jl)

This project is directly and as closely a possible translated from the [Python source code](http://www.neuroimaging.at/pages/qsm.php) (Cython)  
Note: The parallel CPU version is currently about 30% slower than the Cython version, the GPU version is about 5x faster than the Cython version (on a RTX 3060 Laptop GPU 6GB)  

## Setup

1. Install [Julia](https://julialang.org/downloads/) (v1.6 or newer, v1.9+ recommended)
2. Install this package  
    Open the julia REPL and type:

    ```julia
        import Pkg
        Pkg.add(url="https://github.com/korbinian90/TGV_QSM.jl")
        Pkg.add("MriResearchTools") # for nifti handling
    ```

## Example to run TGV

1. Prepare files  
    `mask` and `phase` are required
2. Run this in the REPL or a julia file

    ```julia
        using TGV_QSM, MriResearchTools
        mask = niread("<mask-path>") .!= 0; # convert to boolean
        
        res = [1, 1, 1] # in [mm]
        TE = 0.004 # in [s]
        fieldstrength = 3 # in [T]
        laplace_phi0 = get_laplace_phase3(phase, res); # identical to Python
        # laplace_phi0 = laplacian(phase, res); # faster; identical in normal phase; different in noise
        
        # Runs parallel on CPU
        @time chi = qsm_tgv(laplace_phi0, mask, res; TE, fieldstrength, alpha=(0.0015, 0.0005), iterations=10);
        savenii(chi, "chi", "<folder-to-save>")
    ```

    ```julia
        # Run on GPU
        @time chi = qsm_tgv(laplace_phi0, mask, res; TE, fieldstrength, alpha=(0.0015, 0.0005), iterations=10, gpu=true);
    ```

## Self contained example to test if everything runs

    ```julia
        sz = (20, 20, 20)
        phase = randn(sz)
        laplace_phi0 = get_laplace_phase3(phase)
        mask = trues(sz)
        res = [1, 1, 1]
        omega = [0, 0, 1]
        
        iterations=10
        chi = qsm_tgv(laplace_phi0, mask, res, omega; iterations)
    ```

## Note

Currently the package CUDA.jl is added as dependency. It might be that TGV_QSM.jl can only be downloaded when a NVidia GPU is available. This will be tested in future and made optional in case.
