cd(@__DIR__)
original_path = abspath(".")
p = abspath(joinpath("..", "data", "small"))
tmpdir = mktempdir()
cd(tmpdir)

phasefile_me = joinpath(p, "Phase.nii")
phasefile_me_nan = joinpath(p, "phase_with_nan.nii")
magfile_me = joinpath(p, "Mag.nii")
phasefile_1eco = joinpath(tmpdir, "Phase.nii")
phasefile_2D = joinpath(tmpdir, "Phase2D.nii")
magfile_1eco = joinpath(tmpdir, "Mag.nii")
magfile_2D = joinpath(tmpdir, "Mag2D.nii")
phasefile_1arreco = joinpath(tmpdir, "Phase.nii")
magfile_1arreco = joinpath(tmpdir, "Mag.nii")
maskfile = joinpath(tmpdir, "Mask.nii")
savenii(niread(magfile_me)[:, :, :, 1] |> I -> I .> MriResearchTools.median(I), maskfile)
savenii(niread(phasefile_me)[:, :, :, 1], phasefile_1eco)
savenii(niread(magfile_me)[:, :, :, 1], magfile_1eco)
savenii(niread(phasefile_me)[:, :, :, [1]], phasefile_1arreco)
savenii(niread(magfile_me)[:, :, :, [1]], magfile_1arreco)
savenii(niread(phasefile_me)[:, :, 1, 1], phasefile_2D)
savenii(niread(magfile_me)[:, :, 1, 1], magfile_2D)

phasefile_me_5D = joinpath(tmpdir, "phase_multi_channel.nii")
magfile_5D = joinpath(tmpdir, "mag_multi_channel.nii")
savenii(repeat(niread(phasefile_me), 1, 1, 1, 1, 2), phasefile_me_5D)
savenii(repeat(niread(magfile_me), 1, 1, 1, 1, 2), magfile_5D)

function test_tgv_qsm(args)
    folder = tempname()
    args = [args..., "-o", folder, "-n", "10"]
    @show args
    try
        msg = QuantitativeSusceptibilityMappingTGV_main(args)
        @test msg == 0
        @test isfile(joinpath(folder, "qsm.nii"))
    catch e
        println(args)
        println(sprint(showerror, e, catch_backtrace()))
        @test "test failed" == "with error" # signal a failed test
    end
end

configurations_se(pf, mf) = vcat(configurations_se.([["-p", pf], ["-p", pf, "-m", mf]])...)
configurations_se(pm) = [
    [pm...],
    [pm..., "-N"],
    # [pm..., "--gpu"],
    [pm..., "-i"],
    [pm..., "-f", "7.0"],
    [pm..., "-r", "0"],
    [pm..., "-a", "0.2", "0.5"],
    [pm..., "--threshold", "4"],
    [pm..., "-k", "robustmask"],
    [pm..., "-k", "nomask"],
    [pm..., "-k", "qualitymask"],
    [pm..., "-k", "qualitymask", "0.7"],
]
configurations_me(phasefile_me, magfile_me) = vcat(configurations_me.([["-p", phasefile_me], ["-p", phasefile_me, "-m", magfile_me]])...)
configurations_me(pm) = [
    [pm..., "-e", "1:2", "-t", "[2,4]"], # giving two echo times for two echoes used out of three
    [pm..., "-e", "[1,3]", "-t", "[2,4,6]"], # giving three echo times for two echoes used out of three
    [pm..., "-e", "[1", "3]", "-t", "[2,4,6]"],
    [pm..., "-t", "[2,4,6]"],
    [pm..., "-t", "2:2:6"],
    [pm..., "-t", "[2.1,4.2,6.3]"],
    [pm..., "-t", "epi"], # shorthand for "ones(<num-echoes>)"
    [pm..., "-t", "epi", "5.3"], # shorthand for "5.3*ones(<num-echoes>)"
    [pm..., "-B", "-t", "[2,4,6]"],
    [pm..., "-B", "-t", "[2", "4", "6]"], # when written like [2 4 6] in command line
    [pm..., "--template", "1", "-t", "[2,4,6]"],
    [pm..., "--template", "3", "-t", "[2,4,6]"],
    [pm..., "--phase-offset-correction", "-t", "[2,4,6]"],
    [pm..., "--phase-offset-correction", "bipolar", "-t", "[2,4,6]"],
    [pm..., "--phase-offset-correction", "-t", "[2,4,6]", "--phase-offset-smoothing-sigma-mm", "[5,8,4]"],
]

files = [(phasefile_1eco, magfile_1eco), (phasefile_1arreco, magfile_1arreco), (phasefile_1eco, magfile_1arreco), (phasefile_1arreco, magfile_1eco), (phasefile_2D, magfile_2D)]
for (pf, mf) in files[[1]], args in configurations_se(pf, mf)[[1]]
    test_tgv_qsm(args)
end
for args in configurations_me(phasefile_me, magfile_me)
    test_tgv_qsm(args)
end
for args in configurations_se(["-p", phasefile_me, "-m", magfile_me, "-t", "[2,4,6]"])
    test_tgv_qsm(args)
end
for args in configurations_me(phasefile_me_5D, magfile_5D)[end-2:end] # test the last 3 configurations_me lines for coil combination
    test_tgv_qsm(args)
end
files_se = [(phasefile_1eco, magfile_1eco), (phasefile_1arreco, magfile_1arreco)]
for (pf, mf) in files_se
    b_args = ["-B", "-t", "3.06"]
    test_tgv_qsm(["-p", pf, b_args...])
    test_tgv_qsm(["-p", pf, "-m", mf, b_args...])
end

test_tgv_qsm(["-p", phasefile_me_nan, "-t", "[2,4]", "-k", "nomask"])

## Test error and warning messages
m = "multi-echo data is used, but no echo times are given. Please specify the echo times using the -t option."
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", tmpdir, "-v"])

m = "masking option '0.8' is undefined (Maybe '-k qualitymask 0.8' was meant?)"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", tmpdir, "-v", "-k", "0.8"])

m = "masking option 'blub' is undefined"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", tmpdir, "-v", "-k", "blub"])

m = "Phase offset determination requires all echo times! (2 given, 3 required)"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me_5D, "-o", tmpdir, "-v", "-t", "[1,2]", "-e", "[1,2]", "--phase-offset-correction"])

m = "echoes=[1,5]: specified echo out of range! Number of echoes is 3"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", tmpdir, "-v", "-t", "[1,2,3]", "-e", "[1,5]"])

m = "echoes=[1,5} wrongly formatted!"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", tmpdir, "-v", "-t", "[1,2,3]", "-e", "[1,5}"])

m = "Number of chosen echoes is 2 (3 in .nii data), but 5 TEs were specified!"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", tmpdir, "-v", "-t", "[1,2,3,4,5]", "-e", "[1,2]"])

m = "size of magnitude and phase does not match!"
@test_throws ErrorException(m) QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", tmpdir, "-v", "-t", "[1,2,3]", "-m", magfile_1eco])

m = "robustmask was chosen but no magnitude is available. No mask is used!"
@test_logs (:warn, m) match_mode = :any QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", tmpdir])

m = "The echo times 1 and 2 ([1.1, 1.1, 1.1]) need to be different for MCPC-3D-S phase offset correction! No phase offset correction performed"
@test_logs (:warn, m) match_mode = :any QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-m", magfile_me, "-o", tmpdir, "-t", "[1.1, 1.1, 1.1]", "--phase-offset-correction"])

@test_logs QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", tmpdir, "-m", magfile_1eco]) # test that no warning appears

## test maskfile
test_tgv_qsm(["-p", phasefile_1eco, "-k", maskfile])

## test TGV_QSM output files
println("test TGV_QSM output files")
testpath = joinpath(tmpdir, "test_name_1")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", testpath])
@test isfile(joinpath(testpath, "qsm.nii"))

testpath = joinpath(tmpdir, "test_name_2")
fn = joinpath(testpath, "unwrap_name.nii")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", fn])
@test isfile(fn)

testpath = joinpath(tmpdir, "test_name_2")
gz_fn = joinpath(testpath, "unwrap_name.nii.gz")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_1eco, "-o", gz_fn])
@test isfile(gz_fn)

## test .gz input file
println("test .gz input file")
QuantitativeSusceptibilityMappingTGV_main(["-p", gz_fn, "-o", joinpath(testpath, "gz_read_test.nii")])
QuantitativeSusceptibilityMappingTGV_main(["-p", gz_fn, "-m", gz_fn, "-o", joinpath(testpath, "gz_read_test.nii")])

## test mcpc3ds output files
println("test mcpc3ds output files")
testpath = joinpath(tmpdir, "test5d")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me_5D, "-o", testpath, "-m", magfile_5D, "-t", "[2,4,6]", "-s"])
@test isfile(joinpath(testpath, "combined_mag.nii"))
@test isfile(joinpath(testpath, "combined_phase.nii"))

testpath = joinpath(tmpdir, "test4d")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", testpath, "-m", magfile_me, "-t", "[2,4,6]", "--phase-offset-correction", "-s"])
@test isfile(joinpath(testpath, "corrected_phase.nii"))

## test B0 output files
println("test B0 output files")
testpath = joinpath(tmpdir, "testB0_1")
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", testpath, "-m", magfile_me, "-t", "[2,4,6]", "-B"])
@test isfile(joinpath(testpath, "B0.nii"))

testpath = joinpath(tmpdir, "testB0_2")
name = "B0_output"
QuantitativeSusceptibilityMappingTGV_main(["-p", phasefile_me, "-o", testpath, "-m", magfile_me, "-t", "[2,4,6]", "-B", name])
@test isfile(joinpath(testpath, "$name.nii"))

cd(original_path)
GC.gc()
rm(tmpdir, recursive=true)
