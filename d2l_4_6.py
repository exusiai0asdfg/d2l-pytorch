import torch
from torch import nn

device = torch.device("cuda")
print(f"Using device: {device}")

from fmnist_load import load_fashion_mnist
train_iter, test_iter = load_fashion_mnist(batch_size =256)

import time

start = time.perf_counter()

# 你的完整训练循环


torch.seed = 42

def dropout_layer(x,dropout):
    p = dropout
    assert 0 <= p <=1
    if p in (0, 1):
        return x if p == 0 else torch.zeros_like(x)
    mask = (torch.rand_like(x) > p).float()
    return mask*x / (1.0-p)

# x = torch.arange(16,dtype = torch.float32).reshape((2,8))
# print(x)
# print(dropout_layer(x,0.))
# print(dropout_layer(x,0.5))
# print(dropout_layer(x,1))




dp1,dp2 = 0.2,0.5
lr =0.03
num_epochs = 30
num_inputs,num_hiddens,num_outputs = 784,256,10

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Dropout(dp1),
                    nn.Linear(256,256),
                    nn.ReLU(),
                    nn.Dropout(dp2),
                    nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        
net.apply(init_weights)
net = net.to(device)


trainer = torch.optim.SGD(net.parameters(), lr=lr)
wd = 0
# trainer = torch.optim.SGD([
#     {
#         "params": [m.weight for m in net if isinstance(m, nn.Linear)],
#         "weight_decay": wd
#     },
#     {
#         "params": [m.bias for m in net if isinstance(m, nn.Linear)],
#         "weight_decay": 0
#     }
#], lr=lr)



loss = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total_samples = 0
            
            
    for x,y in train_iter:
        if torch.rand(()) < 0.5:
            x = 1.0 - x
        x = x.to(device)
        y = y.to(device)
        
        trainer.zero_grad()
        y_hat=net(x)
        l = loss(y_hat,y)
        l.backward()
        trainer.step()
        
        batch_n = x.shape[0]
        total_loss += l.detach()*batch_n
        total_samples += batch_n
        pred = torch.argmax(y_hat,dim=1)
        correct += (y_hat.argmax(dim=1) == y).sum()

        
        
    avg_loss = total_loss.item() / total_samples
    train_acc = correct.item() / total_samples
    print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")
    print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, train_acc = {train_acc:.2%}")
    
    
net.eval()

test_correct = 0
test_samples = 0

with torch.no_grad():
    for x,y in test_iter:
        x = x.to(device)
        y = y.to(device)
        x = 1.0-x
        
        y_hat = net(x)
        label = torch.argmax(y_hat,dim=1)
        test_correct += (label==y).sum().item()
        test_samples += x.shape[0]
        
print(test_correct/test_samples)
        
        
if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print(f"训练耗时: {elapsed:.2f}s")

net.eval()

normal_correct = 0
inverted_correct = 0
total = 0

with torch.no_grad():
    for x, y in test_iter:
        x = x.to(device)
        y = y.to(device)

        pred_normal = net(x).argmax(dim=1)
        pred_inverted = net(1.0 - x).argmax(dim=1)

        normal_correct += (pred_normal == y).sum().item()
        inverted_correct += (pred_inverted == y).sum().item()
        total += y.shape[0]

print(f"Normal:   {normal_correct / total:.2%}")
print(f"Inverted: {inverted_correct / total:.2%}")