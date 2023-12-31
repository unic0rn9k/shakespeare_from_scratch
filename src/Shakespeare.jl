module Shakespeare
    using Test

    @static if get(ENV, "INSTALL", 0) == "true"
        using Pkg
        Pkg.add(["FiniteDiff", "Pickle", "Plots"])
    end

    include("autodiff.jl")
    include("optimizer.jl")
    include("transformer.jl")

    export ADGraph, one_hot, softmax, mse_loss, Adam, optimize!, set!, val, SGD, Δ, Δ!, rename!, cross_entropy, NodeID, Optimizer, cat
end
