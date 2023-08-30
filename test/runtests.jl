using TGV_QSM
using Test
using TestItemRunner

@run_package_tests

@testitem "TGV_QSM.jl" begin
    sz = (20, 20, 20)
    phase = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]
    TE = 1

    iterations = 10
    chi = qsm_tgv(phase, mask, res; TE, iterations)

    @test size(chi) == sz
end

@testitem "GPU" begin
    if TGV_QSM.CUDA.functional()
        sz = (20, 20, 20)
        phase = randn(sz)
        mask = trues(sz)
        res = [1, 1, 1]
        omega = [0, 0, 1]
        TE = 1

        iterations = 10
        chi_cpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=false)
        chi_gpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=true)

        relative_diff(A, B) = sum(abs.(A .- B)) / sum(abs.(B))
        @test relative_diff(Array(chi_gpu), chi_cpu) < 1e-7
    end
end

@testitem "Laplacian" begin
    sz = (20, 20, 20)
    phase = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]
    TE = 1

    iterations = 10
    chi_3 = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase3)
    chi_conv = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase_conv)
    chi_romeo = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase_romeo)

    @test chi_3 != chi_conv
    @test chi_romeo != chi_conv
    @test chi_3 != chi_romeo
end

@testitem "Aqua" begin
    using Aqua
    Aqua.test_ambiguities(TGV_QSM)
    Aqua.test_all(TGV_QSM; ambiguities=false)
end