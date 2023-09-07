using Test
using Shakespeare

@testset "PyTorch comparison" begin
    run(`python3 test.py`)
    include("compare.jl")
end
