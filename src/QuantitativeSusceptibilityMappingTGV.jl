module QuantitativeSusceptibilityMappingTGV

using KernelAbstractions, PaddedViews, ImageMorphology, Rotations, OffsetArrays, StaticArrays, ProgressMeter, Statistics, ImageFiltering, ROMEO, LinearAlgebra, ImageFiltering, FFTW

include("tgv.jl")
include("tgv_helper.jl")
include("laplacian.jl")
include("oblique_stencil.jl")

# The references for the method this package implements, registered next to the
# code. The registry lives in ROMEO, which this package already depends on.
#
# ROMEO gained the registry in 1.5, and the compat bound here stays at "1.0" on
# purpose: this package supports Julia 1.7, while ROMEO 1.5 requires 1.9, so an
# older ROMEO is a legitimate resolution and must not be a load error. When that
# happens the citations simply are not registered, and ROMEO's writer reports a
# reference whose owning package is too old rather than omitting it silently.
function __init__()
    isdefined(ROMEO, :register_citation!) || return
    ROMEO.register_citation!(:tgv,
        """Langkammer, C., Bredies, K., Poser, B.A., Barth, M., Reishofer, G., Fan, A.P., Bilgic, B., Fazekas, F., Mainero, C., Ropele, S., 2015.
           Fast quantitative susceptibility mapping using 3D EPI and total generalized variation.
           NeuroImage 111, 622-630.
           https://doi.org/10.1016/j.neuroimage.2015.02.041""")
    ROMEO.register_citation!(:tgv_original,
        """Bredies, K., Ropele, S., Poser, B.A., Barth, M., Langkammer, C., 2014.
           Single-step quantitative susceptibility mapping using total generalized variation and 3D EPI.
           Proceedings of the 22nd Annual Meeting ISMRM, p. 604.""")
    # Both references are for the one method, so they share a heading and appear
    # together under it. Set through the dict rather than the `label` keyword,
    # which ROMEO only gained in 1.5.1: the compat bound here is deliberately
    # loose (see above), so an older ROMEO must still load, just without labels.
    if isdefined(ROMEO, :LABELS)
        ROMEO.LABELS[:tgv] = "TGV QSM"
        ROMEO.LABELS[:tgv_original] = "TGV QSM"
    end
end

export qsm_tgv, get_laplace_phase3, get_laplace_phase_del, get_laplace_phase_romeo, stencil

end
