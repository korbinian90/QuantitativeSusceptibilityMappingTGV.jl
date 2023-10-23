# QuantitativeSusceptibilityMappingTGV

[![Build Status](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/korbinian90/QuantitativeSusceptibilityMappingTGV.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This project is an improvement of the [Python source code](http://www.neuroimaging.at/pages/qsm.php) in terms of speed, artefacts and ease of use.  

## References

- Bredies, K., Ropele, S., Poser, B. A., Barth, M., & Langkammer, C. (2014). Single-step quantitative susceptibility mapping using total generalized variation and 3D EPI. In Proceedings of the 22nd Annual Meeting ISMRM (Vol. 62, p. 604).

- Langkammer, C., Bredies, K., Poser, B. A., Barth, M., Reishofer, G., Fan, A. P., ... & Ropele, S. (2015). Fast quantitative susceptibility mapping using 3D EPI and total generalized variation. Neuroimage, 111, 622-630.

## Command line usage

### Command line Setup

1. Install [Julia](https://julialang.org/downloads/) (v1.9+ recommended)
2. Make sure `julia` can be executed from the command line
3. Download the single file tgv_qsm.jl (<a id="raw-url" href="https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/blob/main/tgv_qsm.jl">Download tgv_qsm.jl</a>)

### Run script

```bash
julia <folder>/tgv_qsm.jl --help
```

On the first usage, the script will download all dependencies.

### Optional configuration

Under Linux: Make the file executable with `chmod +x tgv_qsm.jl` and run via

```bash
<folder>/tgv_qsm.jl --help
```

## Run in Julia

### Setup

1. Install [Julia](https://julialang.org/downloads/) (v1.9+ recommended)
2. Install this package  
    Open the julia REPL and type:

    ```julia
    julia> ] # enters package manager
    pkg> add QuantitativeSusceptibilityMappingTGV MriResearchTools
    ```

    or

    ```julia
    import Pkg
    Pkg.add(["QuantitativeSusceptibilityMappingTGV", "MriResearchTools"])
    ```

## Example to run TGV

1. Prepare files  
    3D `mask` and `phase` NIfTI files are required
2. Run the following commands in an interactive julia REPL or a julia source file

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
    # Change regularization strength (1-4)
    chi = qsm_tgv(phase, mask, res; TE, fieldstrength, regularization=1);
    ```

    ```julia
    # Optionally obtain settings from BIDS JSON file
    using JSON
    settings = JSON.parse(read("<phase-json>", String))
    fieldstrength = settings["MagneticFieldStrength"]
    TE = settings["EchoTime"]
    ```

The first execution might take some time to compile the kernels (~1min).

## Settings

The default settings were optimized for brain QSM and should lead to good results independently of the acquired resolution.

```julia
# Run on CPU in parallel
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, gpu=false);
```

It uses the number of CPU threads julia was started with. You can use `julia --threads=auto` or set it to a specific number of threads.

### Other optional keyword arguments

`erosions=3`: number of erosion steps applied to the mask  
`B0_dir=[0, 0, 1]`: direction of the B0 field. Right angle rotation are currently supported  
`alpha=[0.003, 0.001]`: manually choose regularization parameters (overwrites `regularization`)  
`step_size=3`: requires less iterations with higher step size, but might lead to artefacts or iteration instabilities. `step_size=1` is the behaviour of the publication  
`iterations=1000`: manually choose the number of iteration steps  
`dedimensionalize=false`: optionally apply a dedimensionalization step  
`laplacian=get_laplace_phase_del`: options are `get_laplace_phase3` (corresponds to Cython version), `get_laplace_phase_romeo` and `get_laplace_phase_del` (default). `get_laplace_phase_del` is selected as default as it improves the robustness when imperfect phase values are present  
`correct_laplacian=true`: subtracts the mean of the Laplacian inside the mask, which avoids edge artefacts in certain cases

## Rotated Data

This implementation doesn't support data with an oblique angle acquisition yet. For rotated data, it is recommended to use the [QSMxT pipeline](https://qsmxt.github.io/QSMxT/) for susceptibility mapping, which applies TGV internally

## Self contained example to test if package works

```julia
using QuantitativeSusceptibilityMappingTGV

sz = (20, 20, 20);
res = [1, 1, 1];
TE = 0.01;

phase = randn(sz);
mask = trues(sz);

chi = qsm_tgv(phase, mask, res; TE);
```

## Settings to reproduce the original version

Beside one regularization [bug-fix](https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/commit/0dfe717a09fa766153a3c216243655a30b1359b0), this should produce identical results to the [original Cython code](https://www.neuroimaging.at/pages/qsm.php)

```julia
qsm = qsm_tgv(phase, mask, res; TE, fieldstrength, laplacian=get_laplace_phase3, step_size=1, iterations=1000, alpha=[0.003, 0.001], erosions=5, dedimensionalize=false, correct_laplacian=false)
```

## Speed

The parallel CPU version is about twice as fast as the Cython version, the GPU version is about 10x faster than the Cython version (on a RTX 3060 Laptop GPU 6GB)  

## Run with other GPU types

Other GPU types don't work with the command line script. They have to be accessed via Julia (or the command line script modified).

```julia
using AMDGPU
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, gpu=AMDGPU);
```

For Intel GPU:

```julia
using oneAPI
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, gpu=oneAPI);
```

For Mac GPU:

```julia
using Metal
chi = qsm_tgv(phase, mask, res; TE, fieldstrength, gpu=Metal);
```

## Masking for QSM

Masking for QSM is a challenge, since including corrupted phase areas can cause global artefacts in the QSM result. See the publications [QSMxT](https://doi.org/10.1002%2Fmrm.29048) and [phase based masking](https://doi.org/10.1002/mrm.29368).

A simple solution is to remove areas based on phase quality using [ROMEO](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28563)

```julia
using MriResearchTools
mask_phase = robustmask(romeovoxelquality(phase; mag))
mask_combined = mask_phase .& mask_brain
```

The mask might contain holes, and a more sophisticated approach is taken in the [QSMxT toolbox](https://qsmxt.github.io/QSMxT/).
