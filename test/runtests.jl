using Test
using Shakespeare

@testset "PyTorch comparison" begin
    include("compare.jl")
    #println(keys(models))
    #for name in keys(models)
    #    @info("Building artifacts for '$name'")
    #    run(`python3 py/$name.py`)
    #end
    compare()
end
