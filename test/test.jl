include("../src/Shakespeare.jl")
using .Shakespeare
import Pkg
Pkg.add("Pickle")
Pkg.add("Plots")
using Test
using Pickle
using Plots

mutable struct ModelComparitor
    name::String
    parameters::Dict{String, NodeID}
    optimizer::Union{Optimizer,Nothing}
    iter::Int
    drift::Array{Float64}
    function ModelComparitor(name::String)
        return new(name, Dict(), nothing, 0, [])
    end
end

function Base.:push!(m::ModelComparitor, p::Vector{String})::Vector{NodeID}
    nodes = []
    for p in p
        m.parameters[p] = push!(g, Pickle.Torch.THload("artifacts/$(m.name)_$p$(m.iter).pt"))
        push!(nodes, m.parameters[p])
    end
    nodes
end

function initialize_optimizer!(m::ModelComparitor, optimizer::Optimizer)
    m.optimizer = optimizer
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

g = ADGraph()

models::Dict{String, Function} = Dict(
    "linear_softmax" => function(m)
        (w, x, y, b) = push!(m, ["w", "x", "y", "b"])
        y_hat = softmax(x*w+b)
        loss = mse_loss(y, y_hat)
        optimizer = SGD(0.1, [w], loss)
        initialize_optimizer!(m, optimizer)
        m
    end,
    "adem" => function(m)
        (w, x, b, y) = push!(m, ["w", "x", "b", "y"])
        y_hat = x*w+b
        loss = mse_loss(y, y_hat)
        optimizer = Adam(0.1, [w, b], loss)
        initialize_optimizer!(m, optimizer)
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
            print("Validating '$name'... $(i*100/ntest)%\r")
        end
        saveplot(model)
        delete!(models, name)
        @test model.drift[end] < 0.1
    end
end
