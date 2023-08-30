using MLDatasets

include("autodiff.jl")
include("optimizer.jl")
include("loss.jl")

# Before training: 10.26%
# 100%
# After training:  74.85000000000001%

begin
    mnist = MNIST()

    g = ADGraph()
    w = rand(g, (28 * 28, 10))
    b = rand(g, (1, 10))

    x = push!(g, reshape(mnist.features[:, :, 1], 1, 28 * 28))
    y = push!(g, one_hot((1, mnist.targets[1]), (1, 10)))

    ŷ = softmax(x * w + b)

    loss = cross_entropy(y, ŷ)
    optimizer = Adam(0.001, [w, b], loss)

    nval   = 10000
    ntrain = 20000

    model_accuracy = () -> begin
        correct = 0
        for _ in 1:nval
            i = rand(55001:60000)
            set!(x, reshape(mnist.features[:, :, i], 1, 28 * 28))
            set!(y, one_hot((1, mnist.targets[i]), (1, 10)))
            correct += argmax(val(ŷ)) == argmax(val(y))
        end
        return correct / nval
    end

    println("Before training: $(100*model_accuracy())%\n")

    for i in 0:ntrain
        optimize!(optimizer)
        n = rand(1:55000)
        set!(x, reshape(mnist.features[:, :, n], 1, 28 * 28))
        set!(y, one_hot((1, mnist.targets[n]), (1, 10)))
        println("\u1b[1F$(floor(Int, i*100/ntrain))%")
    end

    println("After training:  $(100*model_accuracy())%")
end
