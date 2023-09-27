# QuantitativeSusceptibilityMappingTGV

[![Build Status](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This project is an improvement of the [Python source code](http://www.neuroimaging.at/pages/qsm.php) in terms of speed, artefacts and ease of use.  

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
    # Optionally obtain settings from BIDS JSON file
    using JSON
    settings = JSON.parse(read("<phase-json>", String))
    fieldstrength = settings["MagneticFieldStrength"]
    TE = settings["EchoTime"]
    ```

The first execution might take some time to compile the kernels (~1min).

## Command Line Interface

Coming soon

## Settings

```julia
# Run on CPU in parallel
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, gpu=false);
```

It uses the number of CPU threads julia was started with. You can use `julia --threads=auto` or set it to a specific number of threads.

```julia
# Change regularization strength (1-4)
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, regularization=1);
```

### Other optional keyword arguments

`erosions=3`: number of erosion steps applied to the mask  
`B0_dir=[0, 0, 1]`: direction of the B0 field. right angle rotation are currently supported  
`alpha=[0.003, 0.001]`: manually choose regularization parameters (overwrites `regularization`)  
`step_size=3`: requires less iterations with highes step size, but might lead to artefacts or iteration instabilities. `step_size=1` is the behaviour of the publication.  
`iterations=1000`: manually choose the number of iteration steps  
`dedimensionalize=false`: optionally apply a dedimensionalization step
`laplacian=get_laplace_phase_del`: options are `get_laplace_phase3` (corresponds to Cython version), `get_laplace_phase_romeo` and `get_laplace_phase_del` (default). `get_laplace_phase_del` is selected as default as it improves robustness with imperfect phase values.  
`correct_laplacian=true`: subtracts the mean of the Laplacian inside the mask, which avoid edge artefacts in certain cases

## Rotated Data

This implementation doesn't support data with a oblique angle acquisition yet. For rotated data, it is recommended to use the [QSMxT pipeline](https://qsmxt.github.io/QSMxT/) for susceptibility mapping, which applies TGV internally

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

## Speed

The parallel CPU version is about twice as fast as the Cython version, the GPU version is about 10x faster than the Cython version (on a RTX 3060 Laptop GPU 6GB)  
