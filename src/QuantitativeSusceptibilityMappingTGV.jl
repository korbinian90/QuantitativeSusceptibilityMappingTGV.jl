module QuantitativeSusceptibilityMappingTGV

using KernelAbstractions, PaddedViews, Rotations, OffsetArrays, StaticArrays, ProgressMeter, Statistics, ImageFiltering, ROMEO, LinearAlgebra, ImageFiltering, FFTW

# Baked in at precompile time; include_dependency so a version bump invalidates
# the cache. A compiled program has no Project.toml to read at run time.
const PKG_VERSION = let toml = joinpath(@__DIR__, "..", "Project.toml")
    include_dependency(toml)
    m = match(r"^version\s*=\s*\"([^\"]+)\""m, read(toml, String))
    m === nothing && error("no version field in $toml")
    VersionNumber(m.captures[1])
end

include("tgv.jl")
include("tgv_helper.jl")
include("laplacian.jl")
include("oblique_stencil.jl")

export qsm_tgv, get_laplace_phase3, get_laplace_phase_del, get_laplace_phase_romeo, stencil

end
