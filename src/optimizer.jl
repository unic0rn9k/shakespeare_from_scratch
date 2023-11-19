# Implementation of the Adam optimizer (written by copilot, for now)
abstract type Optimizer end

struct SGD <: Optimizer
    lr::Real # Learning rate of the optimizer
    params::Array{NodeID} # Parameters to optimize
    loss::NodeID # Loss function to minimize

    function SGD(lr::Real, params::Array{NodeID}, loss::NodeID)
        return new(lr, params, loss)
    end
end

function optimize!(optimizer::SGD)
    for p in optimizer.params
        set!(p, val(p) .- optimizer.lr .* val(Δ(optimizer.loss, p)))
    end
end

mutable struct Adam <: Optimizer
    lr::Real # Learning rate of the optimizer
    params::Array{NodeID} # Parameters to optimize
    loss::NodeID # Loss function to minimize
    β1::Real # Decay rate of the first moment
    β2::Real # Decay rate of the second moment
    ϵ::Real # Small value to prevent division by zero
    t::Int # Current timestep
    m::Array{Array{Float64}} # First moment
    v::Array{Array{Float64}} # Second moment

    function Adam(lr::Real, params::Array{NodeID}, loss::NodeID; β1::Real=0.9, β2::Real=0.999, ϵ::Real=1e-8)
        t = 0
        m = [zeros(size(val(p))) for p in params]
        v = [zeros(size(val(p))) for p in params]
        return new(lr, params, loss, β1, β2, ϵ, t, m, v)
    end
end

function optimize!(optimizer::Adam)
    optimizer.t += 1
    for i in 1:length(optimizer.params)
        p = optimizer.params[i]
        g = val(Δ(optimizer.loss, p))
        optimizer.m[i] = optimizer.β1 * optimizer.m[i] + (1 - optimizer.β1) * g
        optimizer.v[i] = optimizer.β2 * optimizer.v[i] + (1 - optimizer.β2) * g .^ 2
        m̂ = optimizer.m[i] / (1 - optimizer.β1^optimizer.t)
        v̂ = optimizer.v[i] / (1 - optimizer.β2^optimizer.t)
        oops = val(p) .- m̂ .* optimizer.lr ./ (sqrt.(v̂) .+ optimizer.ϵ)
        if isnan.(oops) |> any
            @error "NaN encountered in Adam optimizer"
        end
        if isinf.(oops) |> any
            @error "Inf encountered in Adam optimizer"
        end
        set!(p, oops)
    end
end

