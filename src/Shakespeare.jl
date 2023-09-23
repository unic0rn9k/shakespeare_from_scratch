module Shakespeare
    using Test

    include("autodiff.jl")
    include("optimizer.jl")
    include("loss.jl")

    @testset "PyTorch comparison" begin
        include("test.jl")
        for name in keys(models)
            @info("Building artifacts for '$name'")
            run(`python3 py/$name.py`)
        end
        compare()
    end
end
