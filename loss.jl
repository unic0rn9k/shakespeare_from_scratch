MSE = (y, ŷ) -> sum((y - ŷ)^2)

cross_entropy = (y, ŷ) -> push!(g, 0) - sum(elemmul(y, log(ŷ)))

one_hot = (x, n) -> [x == i ? 1.0 : 0.0 for i in n]