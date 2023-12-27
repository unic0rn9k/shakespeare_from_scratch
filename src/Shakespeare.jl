module Shakespeare
    using Test

    @static if get(ENV, "INSTALL", 0) == "true"
        using Pkg
        Pkg.add(["FiniteDiff", "Pickle", "Plots", "MLDatasets"])
    end

    include("autodiff.jl")
    include("optimizer.jl")
    include("loss.jl")

    @static if get(ENV, "INSTALL", 0) != "true"
        include("transformer.jl")
    end

    export ADGraph, one_hot, softmax, mse_loss, Adam, optimize!, set!, val, SGD, Δ, Δ!, rename!, cross_entropy, NodeID, Optimizer, cat
end
