using MLDatasets: MNIST
include("../src/Shakespeare.jl")
using .Shakespeare

# Instead of going directly to numerically validating the gradients,
# validate each of the operations in the forward pass, and perhaps the entire graph.

## Forward validation
# - [x] 

function classifier()
    mnist = MNIST()

    g = ADGraph()
    w = push!(g, randn(28 * 28, 10))
    b = push!(g, randn(1, 10))

    x = push!(g, randn(1, 28*28))
    y = push!(g, randn(1, 10))
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
    ntrain = 40000

    @show Δ(loss, w)

    model_accuracy = () -> begin
        correct = 0
        closs = 0
        for i in 1:nval
            input!(i)
            target!(i)
            #@show (argmax(val(ŷ)).I[2]-1)
            correct += (argmax(val(ŷ)).I[2]-1) == mnist.targets[i]
            #closs += lossf()
        end
        return correct / nval
    end

    println("Before training: $(100*model_accuracy())%\n")

    for i in 0:ntrain
        n = rand(1:55000)
        input!(n)
        target!(n)
        dw = val(Δ(loss, w))
        db = val(Δ(loss, b))
        ol = val(loss)
        set!(w, val(w) .- (0.01 .* dw))
        set!(b, val(b) .- (0.01 .* db))
        if i%100 == 0
            println("\u1b[1F $i / $ntrain - $(val(loss))  ")
        end
        if val(loss) > ol
            #@warn "$i : rising loss"
        end
    end

    println("After training:  $(100*model_accuracy())%")

    return g, x, ŷ
end

classifier()

#Δ(loss, w) = (0 + ((0 + (T( const X) * ((const Y_hat .* (const 1 - const Y_hat)) .* neg( ((const 2 * const 1) * ^1( (const Y🏷️ - const Y_hat))))))) + 0))
