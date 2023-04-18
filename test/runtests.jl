using TGV_QSM
using Test
using TestItemRunner

@run_package_tests

@testitem "TGV_QSM.jl" begin
    sz = (20, 20, 20)
    laplace_phi0 = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]
    TE = 1

    iterations = 10
    chi = qsm_tgv(laplace_phi0, mask, res; TE, iterations)

    @test size(chi) == sz
end
