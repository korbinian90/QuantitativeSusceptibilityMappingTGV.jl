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
    
    iterations=10
    chi = qsm_tgv(laplace_phi0, mask, res, omega; iterations)

    @test size(chi) == sz
end

@testitem "GPU" begin
    sz = (20, 20, 20)
    laplace_phi0 = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]
    
    iterations=10
    chi = qsm_tgv(laplace_phi0, mask, res, omega; iterations)
    chi_gpu = qsm_tgv(laplace_phi0, mask, res, omega; iterations, gpu=true)
    @test chi_gpu isa TGV_QSM.CuArray

    relative_diff(A, B) = sum(abs.(A .- B)) / sum(abs.(B))
    @test relative_diff(Array(chi_gpu), chi) < 1e-7
end
