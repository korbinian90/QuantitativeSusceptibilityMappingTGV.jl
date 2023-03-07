module TGV_QSM

using CUDA, CUDAKernels, KernelAbstractions, PaddedViews

include("tgv.jl")
include("tgv_helper.jl")

export qsm_tgv

end
