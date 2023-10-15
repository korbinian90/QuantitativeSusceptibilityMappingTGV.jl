import Pkg
Pkg.activate(@__DIR__)
try
    using QuantitativeSusceptibilityMappingTGV, ArgParse
catch
    Pkg.add(["QuantitativeSusceptibilityMappingTGV", "ArgParse"])
    using QuantitativeSusceptibilityMappingTGV, ArgParse
end

@time msg = QuantitativeSusceptibilityMappingTGV_main(ARGS)
println(msg)

# call with julia <path-to-file>/tgv_qsm.jl -p phase.nii.gz -m mask.nii.gz -o output.nii.gz