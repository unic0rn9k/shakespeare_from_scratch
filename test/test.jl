include("../src/Shakespeare.jl")
using .Shakespeare
using Test
using Pickle
using Plots

mutable struct ModelComparitor
    name::String
    parameters::Dict{String, NodeID}
    optimizer::Union{Optimizer,Nothing}
    iter::Int
    drift::Array{Float64}
    graph::ADGraph
    function ModelComparitor(name::String)
        return new(name, Dict(), nothing, 0, [], ADGraph())
    end
end

function Base.:push!(m::ModelComparitor, p::Vector{String})::Vector{NodeID}
    nodes = []
    for p in p
        m.parameters[p] = push!(m.graph, Pickle.Torch.THload("artifacts/$(m.name)_$p$(m.iter).pt"))
        push!(nodes, m.parameters[p])
    end
    nodes
end

function step!(m::ModelComparitor)
    optimize!(m.optimizer)
    m.iter += 1
    drift = 0
    for p in keys(m.parameters)
        w2 = Pickle.Torch.THload("artifacts/$(m.name)_$p$(m.iter).pt")
        if !(m.parameters[p] in m.optimizer.params)
            set!(m.parameters[p], w2)
        else
            drift += sum((val(m.parameters[p]) - w2).^2) / length(w2)
        end
    end
    push!(m.drift, drift / length(keys(m.parameters)))
end

function saveplot(m::ModelComparitor)
    savefig(plot(m.drift), "compare_plots/$(m.name).png")
end

models::Dict{String, Function} = Dict(
    "linear_softmax" => function(m)
        (w, x, y, b) = push!(m, ["w", "x", "y", "b"])
        y_hat = softmax(x*w+b)
        loss = mse_loss(y, y_hat)
        m.optimizer = SGD(0.1, [w], loss)
        m
    end,
    "adem" => function(m)
        (w, x, b, y) = push!(m, ["w", "x", "b", "y"])
        y_hat = x*w+b
        loss = mse_loss(y, y_hat)
        m.optimizer = Adam(0.1, [w, b], loss)
        m
    end
)

ntest = 10000
for (name, f) in models
    @testset "$name" begin
        model = f(ModelComparitor(name))
        for i in 1:ntest
            try
                step!(model)
            catch err
                @warn(err)
                println("Validating '$name'...   done")
                @info("Collected $i samples")
                break
            end
            #print("Validating '$name'... $(i*100/ntest)%\r")
        end
        saveplot(model)
        delete!(models, name)
        @test model.drift[end] < 0.1
    end
end
