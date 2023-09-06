using Pickle
using Plots
include("autodiff.jl")
include("optimizer.jl")
include("loss.jl")

begin
    g = ADGraph()

    w = Pickle.Torch.THload("compare/w0.pt")
    w = push!(g, w)

    x = Pickle.Torch.THload("compare/x0.pt")
    x = push!(g, x)

    loss = sum((x * w) ^ 2)
    optimizer = SGD(0.1, [w], loss)
    drift = []
    
    for i in 0:100
        try
            a = val(w) ≈ Pickle.Torch.THload("compare/w$i.pt")
            b = val(x) ≈ Pickle.Torch.THload("compare/x$i.pt")
            if !a || !b
                @warn("Mismatched values at iteration $i")
            end
            push!(drift, sum((val(w) - Pickle.Torch.THload("compare/w$i.pt")).^2))
            optimize!(optimizer)
            set!(x, Pickle.Torch.THload("compare/x$(i + 1).pt"))
        catch LoadError
            @info("Collected $i samples")
            break
        end
    end

    savefig(plot(drift), "drift.png")
end 
