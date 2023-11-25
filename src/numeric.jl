function numeric_gradient(node::NodeID, wrt::Vector{NodeID}; delta=1e-3, iters=100)::Vector{MathObj}
    grads = [zeros(size(val(i))) for i in wrt]
    y = val(node)

    for _ in 1:iters
        for (n, i) in enumerate(wrt)
            input = deepcopy(i)
            dx = randn(size(val(input))) * delta
            set!(input, val(input) + dx)

            dy = val(NodeID(node.id, input.source)) - y
            for dy in dy
                grads[n] += dy ./ dx
            end

            grads[n] /= length(dy)
        end
    end

    [grad / iters for grad in grads]
end
