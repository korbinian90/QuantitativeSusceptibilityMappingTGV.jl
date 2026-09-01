#!/usr/bin/env -S julia --color=yes --startup-file=no --threads=auto

## Usage

# Call with: `<path-to-file>/tgv_qsm_oneapi.jl ARGS`
# On windows use: `julia --threads=auto <path-to-file>/tgv_qsm_oneapi.jl ARGS`

# Example call:
# `./tgv_qsm_oneapi.jl phase.nii.gz mask.nii.gz --TE 0.025 --output output.nii.gz

import Pkg

## Uncomment to use a local julia package directory instead of the global one
# package_dir = joinpath(@__DIR__, ".tgv_cmd_packages")
# mkpath(package_dir)
# Pkg.activate(package_dir)

try
    using oneAPI, QuantitativeSusceptibilityMappingTGV, MriResearchTools, Comonicon
catch
    Pkg.add(["oneAPI", "QuantitativeSusceptibilityMappingTGV", "MriResearchTools", "Comonicon"])
    using oneAPI, QuantitativeSusceptibilityMappingTGV, MriResearchTools, Comonicon
end

version = Comonicon.get_version(QuantitativeSusceptibilityMappingTGV)
Comonicon.get_version(::Module) = version

# filter(!isempty, ...): splitting on both separators turns "[0, 0, 1]" into
# ["0", "", "0", "", "1"], and parse("") throws. ';' is accepted too, since a
# direction copied out of Matlab arrives as "[0;0;1]".
Base.tryparse(::Type{Array{Float64}}, s) = parse.(Float64, filter(!isempty, split(replace(s, "[" => "", "]" => ""), [',', ' ', ';'])))
Base.tryparse(::Type{DataType}, s) = get(Dict("Float32" => Float32, "Float64" => Float64), s, nothing)

Comonicon.@main function tgv_qsm(fn_phase, fn_mask; TE::Float64, output::String="output.nii.gz", fieldstrength::Float64=3.0, regularization::Float64=2.0, erosions::Int=3, B0_dir::Array{Float64}=[0.0, 0.0, 1.0], dedimensionalize::Bool=false, no_laplacian_correction::Bool=false, step_size::Float64=3.0, type::DataType=Float32, nblocks::Int=32)
    println("Starting calculation...")
    phase = readphase(fn_phase)
    mask = niread(fn_mask) .!= 0
    res = header(phase).pixdim[2:4]
    println("Resolution from NIfTI header [mm]: $(round.(Float64.(res); digits=2))")
    chi = qsm_tgv(phase, mask, res; TE, B0_dir, fieldstrength, regularization, erosions, dedimensionalize, correct_laplacian=!no_laplacian_correction, gpu=oneAPI, step_size, type, nblocks)
    println("Writing output")
    savenii(chi, output; header=header(phase))
end
