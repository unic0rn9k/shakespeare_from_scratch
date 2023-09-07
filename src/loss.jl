mse_loss = (y, ŷ) -> sum((y - ŷ)^2) / push!(g, length(val(ŷ)))

cross_entropy = (y, ŷ) -> -sum(elemmul(y, log(ŷ + push!(g, 1e-8))))

one_hot = (x, n) -> [x == i ? 1.0 : 0.0 for i in Tuple.(CartesianIndices(n))]
