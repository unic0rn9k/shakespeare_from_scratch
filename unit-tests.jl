include("autodiff.jl")
include("optimizer.jl")
include("loss.jl")

begin
    g = ADGraph()
    w = rand(g, (10, 10))
    b = rand(g, (1, 10))
    x = rand(g, (1, 10))

    ŷ = softmax(x * w + b)
    y = push!(g, one_hot(argmax(val(x)), CartesianIndices(CartesianIndex(1, 10))))

    loss = cross_entropy(y, ŷ)
    println("Before training: ", val(loss))

    optimizer = Adam(0.01, [w, b], loss)

    for _ in 0:10
        set!(x, rand(1, 10))
        set!(y, one_hot(argmax(val(x)), 10))
        optimize!(optimizer)
    end

    println("After training: ", val(loss))
end