from torch import nn
import torch
w = nn.Parameter(torch.tensor([2.0]))
print(w)
print(w.shape)
print(w.data)
print(w.grad)

l = w ** 2
pramas = [w]
optimizer = torch.optim.SGD(pramas,lr=0.1)
optimizer.zero_grad()
l.backward()
print(w.item())
print(w.grad.item())
optimizer.step()
print(w.item())