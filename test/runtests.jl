using QuantitativeSusceptibilityMappingTGV
using Test
using TestItemRunner

@run_package_tests

@testitem "QuantitativeSusceptibilityMappingTGV.jl" begin
    sz = (20, 20, 20)
    phase = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    TE = 1

    iterations = 10
    chi = qsm_tgv(phase, mask, res; TE, iterations)

    @test size(chi) == sz
end

@testitem "CPU" begin
    sz = (20, 20, 20)
    phase = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    omega = [0, 0, 1]
    TE = 1

    iterations = 10
    chi = qsm_tgv(phase, mask, res; TE, iterations, gpu=false)

    @test size(chi) == sz
end

@testitem "GPU" begin
    if QuantitativeSusceptibilityMappingTGV.CUDA.functional()
        sz = (20, 20, 20)
        phase = randn(sz)
        mask = trues(sz)
        res = [1, 1, 1]
        TE = 1

        iterations = 10
        chi_cpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=false)
        chi_gpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=true)

        relative_diff(A, B) = sum(abs.(A .- B)) / sum(abs.(B))
        @test relative_diff(Array(chi_gpu), chi_cpu) < 1e-7
    end
end

@testitem "CUDA" begin
    if QuantitativeSusceptibilityMappingTGV.CUDA.functional()
        sz = (20, 20, 20)
        phase = randn(sz)
        mask = trues(sz)
        res = [1, 1, 1]
        TE = 1

        iterations = 10
        chi_cpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=false)
        chi_gpu = qsm_tgv(phase, mask, res; TE, iterations, gpu=QuantitativeSusceptibilityMappingTGV.CUDA)

        relative_diff(A, B) = sum(abs.(A .- B)) / sum(abs.(B))
        @test relative_diff(Array(chi_gpu), chi_cpu) < 1e-7
    end
end

@testitem "Laplacian" begin
    sz = (20, 20, 20)
    phase = randn(sz)
    mask = trues(sz)
    res = [1, 1, 1]
    TE = 1

    iterations = 10
    chi_3 = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase3)
    chi_conv = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase_del)
    chi_romeo = qsm_tgv(phase, mask, res; TE, iterations, laplacian=get_laplace_phase_romeo)

    @test chi_3 != chi_conv
    @test chi_romeo != chi_conv
    @test chi_3 != chi_romeo
end

@testitem "Aqua" begin
    using Aqua
    Aqua.test_ambiguities(QuantitativeSusceptibilityMappingTGV)
    Aqua.test_all(QuantitativeSusceptibilityMappingTGV; ambiguities=false)
end

@testitem "TGV_QSM command line tests" begin
    if VERSION ≥ v"1.9"
        using ArgParse, MriResearchTools
        # include("TGV_QSMApp/command_line.jl")
    end
end