using Pickle
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
    optimizer = SGD(0.01, [w], loss)
    
    for i in 0:3
        a = val(w) ≈ Pickle.Torch.THload("compare/w$i.pt")
        b = val(x) ≈ Pickle.Torch.THload("compare/x$i.pt")
        if !a && !b
            println("Mismatch at $i")
        end
        optimize!(optimizer)
    end
end 
