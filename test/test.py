import torch

w = torch.rand(5, 10, requires_grad=True)

optimizer = torch.optim.Adam([w], lr=0.1)

ntest = 10000

for i in range(ntest):
    x = torch.rand(1,5) * 2 - 1
    y = torch.rand(1,10) * 2 - 1

    torch.save(x, f"compare/x{i}.pt")
    torch.save(y, f"compare/y{i}.pt")
    torch.save(w, f"compare/w{i}.pt")

    optimizer.zero_grad()
    yhat = x.mm(w)
    
    loss = torch.nn.functional.mse_loss(yhat, y)

    loss.backward()
    optimizer.step()

    print(f"Generating python test artifacts... {i*100/ntest}%", end="\r")
print("Generating python test artifacts...   done")
