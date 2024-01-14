mse_loss = (y::NodeID, ŷ) -> sum((y - ŷ)^2) / push!(y.source, length(val(ŷ)))
cross_entropy = (y, ŷ) -> -sum(elemmul(y, log(ŷ + push!(y.source, 1e-8))))
one_hot = (x, n) -> [x == i ? 1.0 : 0.0 for i in Tuple.(CartesianIndices(n))]

abstract type Optimizer end

struct SGD <: Optimizer
    lr::Real # Learning rate of the optimizer
    params::Array{NodeID} # Parameters to optimize
    loss::NodeID # Loss function to minimize

    function SGD(lr::Real, params::Vector{N}, loss::N) where N<:NodeID
        return new(lr, params, loss)
    end
end

function optimize!(optimizer::SGD)
    # Should be split into 2 loops for performance
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
    δ::Array{MathObj}
    Δ::Array{NodeID}

    function Adam(lr::Real, params::Vector{N}, loss::N; β1::Real=0.9, β2::Real=0.999, ϵ::Real=1e-8) where N<:NodeID
        t = 0
        m = [zeros(size(p)) for p in params]
        v = [zeros(size(p)) for p in params]
        δ = [zeros(size(p)) for p in params]
        deltas = [Δ(loss, p) for p in params]
        return new(lr, params, loss, β1, β2, ϵ, t, m, v, δ, deltas)
    end
end

function optimize(optimizer::Adam)
    optimizer.t += 1
    for (i, _) in enumerate(optimizer.params)
        g = val(optimizer.Δ[i])
        optimizer.m[i] = optimizer.β1 * optimizer.m[i] + (1 - optimizer.β1) * g
        optimizer.v[i] = optimizer.β2 * optimizer.v[i] + (1 - optimizer.β2) * g .^ 2
        m̂ = optimizer.m[i] / (1 - optimizer.β1^optimizer.t)
        v̂ = optimizer.v[i] / (1 - optimizer.β2^optimizer.t)
        oops = m̂ .* optimizer.lr ./ (sqrt.(v̂) .+ optimizer.ϵ)

        if isnan.(oops) |> any
            @error "NaN encountered in Adam optimizer"
        end
        if isinf.(oops) |> any
            @error "Inf encountered in Adam optimizer"
        end
        optimizer.δ[i] += oops
    end
end

function update!(optimizer::Adam)
    for (i, p) in enumerate(optimizer.params)
        set!(p, val(p) .- optimizer.δ[i])
        optimizer.δ[i] = zeros(size(p))
    end
end

function optimize!(optimizer::Adam)
    optimize(optimizer)
    update!(optimizer)
end
