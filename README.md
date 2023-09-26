# QuantitativeSusceptibilityMappingTGV

[![Build Status](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This project is an improvement of the [Python source code](http://www.neuroimaging.at/pages/qsm.php) (Cython)  
The parallel CPU version is about twice as fast as the Cython version, the GPU version is about 10x faster than the Cython version (on a RTX 3060 Laptop GPU 6GB)  

## References

- Bredies, K., Ropele, S., Poser, B. A., Barth, M., & Langkammer, C. (2014). Single-step quantitative susceptibility mapping using total generalized variation and 3D EPI. In Proceedings of the 22nd Annual Meeting ISMRM (Vol. 62, p. 604).

- Langkammer, C., Bredies, K., Poser, B. A., Barth, M., Reishofer, G., Fan, A. P., ... & Ropele, S. (2015). Fast quantitative susceptibility mapping using 3D EPI and total generalized variation. Neuroimage, 111, 622-630.

## Setup

1. Install [Julia](https://julialang.org/downloads/) (v1.9+ recommended)
2. Install this package  
    Open the julia REPL and type:

    ```julia
    import Pkg
    Pkg.add(url="https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl")
    Pkg.add("MriResearchTools") # for nifti handling
    ```

## Example to run TGV

1. Prepare files  
    3D `mask` and `phase` NIfTI files are required
2. Run this in the REPL or a julia file

    ```julia
    using QuantitativeSusceptibilityMappingTGV, MriResearchTools

    mask = niread("<mask-path>") .!= 0; # convert to boolean
    phase = readphase("<phase-path>");
    
    voxel_size = header(phase).pixdim[2:4] # in [mm]
    TE = 0.004 # in [s]
    fieldstrength = 3 # in [T]
    
    # Automatically runs on GPU, if a CUDA device is detected
    chi = qsm_tgv(phase, mask, res; TE=TE, fieldstrength=fieldstrength);

    savenii(chi, "chi", "<folder-to-save>")
    ```

    ```julia
    # Run on CPU in parallel
    chi = qsm_tgv(phase, mask, res; TE, fieldstrength, alpha=(0.0015, 0.0005), gpu=false);
    ```

    ```julia
    # Convenient way to obtain settings from JSON file
    using JSON
    settings = JSON.parse(read("<phase-json>", String))
    fieldstrength = settings["MagneticFieldStrength"]
    TE = settings["EchoTime"]
    ```

    It uses the number of CPU threads julia was started with. You can use `julia --threads=auto` or set it to a specific number of threads.

The first execution might take some time to compile the kernels (~1min).

## Self contained example to test if everything works

```julia
using QuantitativeSusceptibilityMappingTGV

sz = (20, 20, 20)
res = [1, 1, 1]
TE = 0.01

phase = randn(sz)
mask = trues(sz)

chi = qsm_tgv(phase, mask, res; TE)
```
