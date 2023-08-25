include("autodiff.jl")
include("optimizer.jl")
include("loss.jl")

begin # Simple perceptron, trying to fit a linear function
    g = ADGraph()
    w = rand(g, (10, 10))
    b = rand(g, (1, 10))
    x = rand(g, (1, 10))

    ŷ = x * w + b
    y = push!(g, val(x) .* 2 .+ 1)

    loss = MSE(y, ŷ)

    println("Before training: ", val(loss))

    optimizer = Adam(0.01, [w, b], loss)

    for _ in 0:10
        set!(x, rand(1, 10))
        set!(y, val(x) .* 2 .+ 1)
        optimize!(optimizer)
    end

    println("After training: ", val(loss))
end