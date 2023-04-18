module TGV_QSM

using CUDA, CUDAKernels, KernelAbstractions, PaddedViews, ImageMorphology, Interpolations, Rotations, OffsetArrays, StaticArrays, ImageFiltering, LinearAlgebra
using ParallelStencil, ParallelStencil.FiniteDifferences3D

if CUDA.functional()
    @eval @init_parallel_stencil(CUDA, Float32, 3)
else
    @eval @init_parallel_stencil(Threads, Float32, 3)
end


# struct PS_Setup{T}
#     device::Symbol
#     type::Type
# end
# PS_Setup(device, type) = PS_Setup{type}(device, type)

# function environment!(model::PS_Setup{T}) where {T}
#     # start ParallelStencil
#     if model.device == :gpu
#         eval(:(@init_parallel_stencil(CUDA, $T, 3)))
#         Base.eval(Main, Meta.parse("using CUDA"))
#     else
#         @eval begin
#             @init_parallel_stencil(Threads, $T, 3)
#         end
#     end

#     # includes and exports
#     @eval begin
#         include(joinpath(@__DIR__, "tgv.jl"))
#         include(joinpath(@__DIR__, "tgv_helper.jl"))
#         include(joinpath(@__DIR__, "resample.jl"))
#         include(joinpath(@__DIR__, "structure_tensor.jl"))
#         export qsm_tgv, get_laplace_phase3, laplacian, rotate3D, st_gauss, erode_mask, use_gpu, use_cpu
#     end
# end

include("tgv.jl")
include("tgv_helper.jl")
include("resample.jl")
include("structure_tensor.jl")

export qsm_tgv, get_laplace_phase3, laplacian, rotate3D, st_gauss, erode_mask, use_gpu, use_cpu

end
