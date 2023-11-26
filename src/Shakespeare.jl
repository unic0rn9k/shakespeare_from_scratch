module Shakespeare
    using Test

    include("autodiff.jl")
    include("optimizer.jl")
    include("loss.jl")

    export ADGraph, one_hot, softmax, mse_loss, Adam, optimize!, set!, val, SGD, Δ, Δ!, rename!, cross_entropy, NodeID, Optimizer
end
