using TGV_QSM
using Test
using TestItemRunner

@run_package_tests

@testitem "TGV_QSM.jl" begin
    sz = (20, 20, 20)
    laplace_phi0 = ones(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]

    chi = qsm_tgv(laplace_phi0, mask, res, omega; alpha=(0.2, 0.1), iterations=2)

    @test size(chi) == sz
end
