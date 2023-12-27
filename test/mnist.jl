using MLDatasets: MNIST
include("../src/Shakespeare.jl")
using .Shakespeare

function classifier()
    mnist = MNIST()

    g = ADGraph()
    w = push!(g, randn(28 * 28, 10))
    b = push!(g, randn(1, 10))

    x = push!(g, zeros(1, 28*28))
    y = push!(g, zeros(1, 10))
    ŷ = softmax(x * w + b)

    input!(i) = set!(x, reshape(mnist.features[:, :, i], 1, 28 * 28))
    target!(i) = set!(y, one_hot((1, mnist.targets[i]+1), (1, 10)))

    rename!(w, "const w")
    rename!(b, "const b")
    rename!(x, "const x")
    rename!(y, "const Y")
    rename!(ŷ, "const ŷ")


    loss = sum((ŷ - y)^2)
    nval   = 1000
    ntrain = 10000

    @show Δ(loss, w)

    model_accuracy = () -> begin
        correct = 0
        for i in 1:nval
            input!(i)
            target!(i)
            correct += (argmax(val(ŷ)).I[2]-1) == mnist.targets[i]
        end
        return correct / nval
    end

    println("Before training: $(100*model_accuracy())%\n")

    optim = Adam(0.01, [w, b], loss)

    for i in 0:ntrain
        n = rand(1:55000)
        input!(n)
        target!(n)
        optimize!(optim)

        if i%100 == 0
            println("\u1b[1F $i / $ntrain - $(val(loss))  ")
        end
    end

    acc = 100*model_accuracy()
    println("After training:  $(acc)%")
    @static if get(ENV, "TEST", 0) == "true"
        @test acc > 80
    end

    return g, x, ŷ
end

classifier()
