include("autodiff.jl")
include("optimizer.jl")
include("loss.jl")

begin
    g = ADGraph()

    w = rand(g, (10, 5))
    b = rand(g, (1, 5))
    x = rand(g, (1, 10))

    rename!(w, "w")
    rename!(b, "b")
    rename!(x, "x")

    ŷ = x * w - b

    loss = sum(ŷ^2)
    optimizer = SGD(0.001, [w, b], loss)

    nval = 10000
    ntrain = 10000

    println("loss = $loss")
    println("Δloss = $(Δ(loss, b))")

    model_accuracy = () -> begin
        loss_sum = 0
        for i in 1:nval
            set!(x, rand(1, 10))
            loss_sum += val(loss)[1]
        end
        return loss_sum/nval
    end

    println("Before training: $(model_accuracy())\n")

    for i in 0:ntrain
        optimize!(optimizer)
        n = rand(1:60000)
        set!(x, rand(1, 10))
        println("\u1b[1F$(i*100/ntrain)%")
    end

    println("After training:  $(model_accuracy())")
end
