# TGV_QSM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://korbinian90.github.io/TGV_QSM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://korbinian90.github.io/TGV_QSM.jl/dev/)
[![Build Status](https://github.com/korbinian90/TGV_QSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/korbinian90/TGV_QSM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/korbinian90/TGV_QSM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/korbinian90/TGV_QSM.jl)

## Setup

1. Install [Julia](https://julialang.org/downloads/) (v1.6 or newer)
2. Install this package  
    Open the julia REPL and type:

    ```julia
        import Pkg;
        Pkg.add("https://github.com/korbinian90/TGV_QSM.jl")
        Pkg.add("MriResearchTools") # for nifti handling
    ```

## Example to run TGV

1. Prepare files  
    `mask` and `laplacian` are required
2. Run this in the REPL or a julia file

    ```julia
        using TGV_QSM, MriResearchTools
        mask = niread("<mask-path>") .!= 0 # convert to boolean
        laplace_phi0 = niread("<laplacian-path>")
        res = [1, 1, 1]
        omega = [0, 0, 1]
        
        @time chi = qsm_tgv(laplace_phi0, mask, res, omega; alpha=(0.0015, 0.0005), iterations=10)
        
        savenii(chi, "chi", "<folder-to-save>")
    ```
