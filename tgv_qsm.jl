#!/usr/bin/env -S julia --color=yes --startup-file=no --threads=auto

## Usage

# Call with: `<path-to-file>/tgv_qsm.jl ARGS`
# On windows use: `julia --threads=auto <path-to-file>/tgv_qsm.jl ARGS`

# Example call:
# `./tgv_qsm.jl phase.nii.gz mask.nii.gz --TE 0.025 --output output.nii.gz

import Pkg

## Uncomment to use a local julia package directory instead of the global one
# package_dir = joinpath(@__DIR__, ".tgv_cmd_packages")
# mkpath(package_dir)
# Pkg.activate(package_dir)

try
    using QuantitativeSusceptibilityMappingTGV, MriResearchTools, Comonicon
catch
    Pkg.add(["QuantitativeSusceptibilityMappingTGV", "MriResearchTools", "Comonicon"])
    using QuantitativeSusceptibilityMappingTGV, MriResearchTools, Comonicon
end

version = Comonicon.get_version(QuantitativeSusceptibilityMappingTGV)
Comonicon.get_version(::Module) = version

@main function tgv_qsm(fn_phase, fn_mask; TE::Float64, output::String="output.nii.gz", fieldstrength::Float64=3.0, regularization::Float64=2.0, erosions::Int=3, B0_dir::Array{Int}=[0,0,1], dedimensionalize::Bool=false, no_laplacian_correction::Bool=false, step_size::Float64=3.0, type::DataType=Float32, nblocks::Int=32)
    println("Starting calculation...")
    phase = readphase(fn_phase)
    mask = niread(fn_mask) .!= 0
    res = header(phase).pixdim[2:4]
    println("Resolution from NIfTI header [mm]: $(round.(Float64.(res); digits=2))")
    chi = qsm_tgv(phase, mask, res; TE, B0_dir, fieldstrength, regularization, erosions, dedimensionalize, correct_laplacian=!no_laplacian_correction, step_size, type, nblocks)
    println("Writing output")
    savenii(chi, output; header=header(phase))
end
