using Pickle
using Plots

begin
    g = ADGraph()

    w = Pickle.Torch.THload("compare/w0.pt")
    w = push!(g, w)

    x = Pickle.Torch.THload("compare/x0.pt")
    x = push!(g, x)

    y = Pickle.Torch.THload("compare/y0.pt")
    y = push!(g, y)

    loss = mse_loss(y, x*w)
    optimizer = Adam(0.1, [w], loss)
    drift = []
    
    ntest = 10000
    for i in 0:ntest
        try
            push!(drift, sum((val(w) - Pickle.Torch.THload("compare/w$i.pt")).^2))
            @test drift[i+1] < 1e-4
            optimize!(optimizer)
            set!(x, Pickle.Torch.THload("compare/x$(i + 1).pt"))
            set!(y, Pickle.Torch.THload("compare/y$(i + 1).pt"))
        catch LoadError
            println("Validating tests...   done")
            @info("Collected $i samples")
            break
        end
        print("Validating tests... $(i*100/ntest)%\r")
    end

    savefig(plot(drift), "drift.png")
end 
