include("autodiff.jl")

MSE = (y, ŷ) -> sum((y - ŷ)^2)