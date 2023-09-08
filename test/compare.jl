using Pickle
using Plots

begin
    g = ADGraph()

    w = Pickle.Torch.THload("artifacts/w0.pt")
    w = push!(g, w)

    x = Pickle.Torch.THload("artifacts/x0.pt")
    x = push!(g, x)

    y = Pickle.Torch.THload("artifacts/y0.pt")
    y = push!(g, y)

    loss = mse_loss(y, softmax(x*w))
    optimizer = SGD(0.1, [w], loss)
    drift = []
    
    ntest = 10000
    for i in 0:ntest
        try
            w2 = Pickle.Torch.THload("artifacts/w$i.pt")
            push!(drift, sum((val(w) - w2).^2)/length(w2))
            @test drift[i+1] < 1e-4
            optimize!(optimizer)
            set!(x, Pickle.Torch.THload("artifacts/x$(i + 1).pt"))
            set!(y, Pickle.Torch.THload("artifacts/y$(i + 1).pt"))
        catch LoadError
            println("Validating tests...   done")
            @info("Collected $i samples")
            break
        end
        print("Validating tests... $(i*100/ntest)%\r")
    end

    savefig(plot(drift), "drift.png")
end 
