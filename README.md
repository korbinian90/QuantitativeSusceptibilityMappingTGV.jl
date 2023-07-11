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
        phase = readphase("<phase-path>");
        
        res = [1, 1, 1] # in [mm]
        TE = 0.004 # in [s]
        fieldstrength = 3 # in [T]
        
        # Automatically runs on GPU, if a CUDA device is detected
        @time chi = qsm_tgv(phase, mask, res; TE, fieldstrength);

        savenii(chi, "chi", "<folder-to-save>")
    ```

    ```julia
        # Run on CPU in parallel
        @time chi = qsm_tgv(phase, mask, res; TE, fieldstrength, alpha=(0.0015, 0.0005), gpu=false);
    ```

    It uses the number of threads julia was started with. You can use `julia --threads=auto` or set it to a specific number of threads.

The first execution might take some time to compile the kernels (<1min).

## Self contained example to test if everything works

```julia
   using TGV_QSM

    sz = (20, 20, 20)
    res = [1, 1, 1]
    TE = 0.01

    phase = randn(sz)
    mask = trues(sz)

    chi = qsm_tgv(phase, mask, res; TE)
```
