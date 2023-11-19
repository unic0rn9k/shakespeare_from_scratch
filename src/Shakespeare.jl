module Shakespeare
    using Test

    include("autodiff.jl")
    include("optimizer.jl")
    include("loss.jl")

    const test_mode = false
    if test_mode
        @testset "PyTorch comparison" begin
            include("test.jl")
            for name in keys(models)
                @info("Building artifacts for '$name'")
                run(`python3 py/$name.py`)
            end
            compare()
        end
    end

    export ADGraph
    export one_hot
    export softmax
    export mse_loss
    export Adam
    export optimize!
    export set!
    export val
    export SGD
    export Δ
    export Δ!
    export rename!
    export cross_entropy
end
