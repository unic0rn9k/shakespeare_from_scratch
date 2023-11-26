mse_loss = (y::NodeID, ŷ) -> sum((y - ŷ)^2) / push!(y.source, length(val(ŷ)))

cross_entropy = (y, ŷ) -> -sum(elemmul(y, log(ŷ + push!(y.source, 1e-8))))

one_hot = (x, n) -> [x == i ? 1.0 : 0.0 for i in Tuple.(CartesianIndices(n))]
