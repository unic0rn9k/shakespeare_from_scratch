using MLDatasets: MNIST
include("../src/Shakespeare.jl")
using .Shakespeare

# Instead of going directly to numerically validating the gradients,
# validate each of the operations in the forward pass, and perhaps the entire graph.

## Forward validation
# - [x] 

function classifier()

    g = ADGraph()
    w = push!(g, 2)
    b = push!(g, 1)

    x = push!(g, randn())
    ŷ = x * w + b

    y = push!(g, val(x) * 4 - 2)
    loss = sum((ŷ - y)^2)

    rename!(w, "const w")
    rename!(b, "const b")
    rename!(x, "const x")
    rename!(y, "const Y")
    rename!(ŷ, "const ŷ")

    @show Δ(loss, w)
    @show Δ(loss, b)
    @show val(w) - val(Δ(loss, w))
    @show val(b) - val(Δ(loss, b))
    @show val(x)
    @show val(loss)

    iter = 0
    while val(loss) > 0.1
        set!(x, randn())
        set!(y, val(x) * 4 - 2)
        set!(w, val(w) - val(Δ(loss, w))*0.1)
        set!(b, val(b) - val(Δ(loss, b))*0.1)
        iter+=1
    end
    @show iter
end

mnist = MNIST()
classifier()

#Δ(loss, w) = (0 + ((0 + (T( const X) * ((const Y_hat .* (const 1 - const Y_hat)) .* neg( ((const 2 * const 1) * ^1( (const Y🏷️ - const Y_hat))))))) + 0))
