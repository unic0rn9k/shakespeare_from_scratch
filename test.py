import torch
import pickle

# Pickle the dictionary
#with open("test.pickle", "wb") as f:
#    pickle.dump(torch.rand(1,5), f)

w = torch.rand(5, 10, requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.1)

for i in range(100):
    x = torch.rand(1,5)
    torch.save(x, f"compare/x{i}.pt")
    torch.save(w, f"compare/w{i}.pt")
    optimizer.zero_grad()
    y = x.mm(w)
    loss = (y**2).sum()
    loss.backward()
    optimizer.step()
