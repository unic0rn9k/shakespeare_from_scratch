import torch
import torch.nn.functional as F


def onehot(x, n):
    y = torch.zeros(n)
    y[x] = 1
    return y


w = torch.tensor(torch.rand(10, 10) * 2 - 1, requires_grad=True)
b = torch.tensor(torch.rand(1, 10) * 2 - 1, requires_grad=True)

x = torch.rand(1, 10)
y = onehot(int(x.argmax()), 10)

out = torch.softmax(torch.matmul(x, w) + b, dim=1)

loss = F.cross_entropy(out, x.argmax().unsqueeze(0))
print(float(loss))

optimizer = torch.optim.SGD([w, b], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    loss = F.cross_entropy(out, x.argmax().unsqueeze(0))
    optimizer.step()
    x = torch.rand(1, 10)
print(float(loss))
