module TGV_QSM

using CUDA, CUDAKernels, KernelAbstractions, PaddedViews, ImageMorphology, Interpolations, Rotations, OffsetArrays, StaticArrays
using ParallelStencil, ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float32, 3)

include("tgv.jl")
include("tgv_helper.jl")
include("resample.jl")

export qsm_tgv, get_laplace_phase3, laplacian, rotate3D

end
