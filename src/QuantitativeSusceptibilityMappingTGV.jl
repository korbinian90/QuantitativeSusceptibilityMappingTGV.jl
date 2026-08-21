module QuantitativeSusceptibilityMappingTGV

using KernelAbstractions, PaddedViews, ImageMorphology, Interpolations, Rotations, OffsetArrays, StaticArrays, ProgressMeter, Statistics, ImageFiltering, ROMEO, LinearAlgebra, ImageFiltering, FFTW

include("tgv.jl")
include("tgv_helper.jl")
include("laplacian.jl")
include("oblique_stencil.jl")

# The references for the method this package implements, registered next to the
# code. The registry lives in ROMEO, which this package already depends on.
function __init__()
    ROMEO.register_citation!(:tgv,
        """Langkammer, C., Bredies, K., Poser, B.A., Barth, M., Reishofer, G., Fan, A.P., Bilgic, B., Fazekas, F., Mainero, C., Ropele, S., 2015.
           Fast quantitative susceptibility mapping using 3D EPI and total generalized variation.
           NeuroImage 111, 622-630.
           https://doi.org/10.1016/j.neuroimage.2015.02.041""")
    ROMEO.register_citation!(:tgv_original,
        """Bredies, K., Ropele, S., Poser, B.A., Barth, M., Langkammer, C., 2014.
           Single-step quantitative susceptibility mapping using total generalized variation and 3D EPI.
           Proceedings of the 22nd Annual Meeting ISMRM, p. 604.""")
end

export qsm_tgv, get_laplace_phase3, get_laplace_phase_del, get_laplace_phase_romeo, stencil

end
