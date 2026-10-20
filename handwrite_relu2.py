import torch 
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dataset = datasets.FashionMNIST(
    root="../data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.FashionMNIST(
    root="../data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_iter = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=0
)

test_iter = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

num_inputs = 784
num_hiddens = 256
num_outputs = 10
lr = 0.3
epochs = 15

w1 = nn.Parameter(torch.randn(num_inputs,num_hiddens)*0.01)
b1 = nn.Parameter(torch.zeros_like(num_hiddens))
w2 = nn.Parameter(torch.randn(num_hiddens,num_outputs)*0.01)
b2 = nn.Parameter(torch.zeros_like(num_inputs))

pramas = [w1,w2,b1,b2]
optimizer = torch.optim.SGD(pramas,lr=lr)
loss_fn  = nn.CrossEntropyLoss()

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(a,x)

def net(x):
    x = x.reshape(-1,num_inputs)
    h = x@w1+b1
    o = relu(h)@w2+b2
    return o



for epoch in range(epochs):
    total_loss = 0
    total_samples = 0
    correct = 0
    for x,y in train_iter:
        optimizer.zero_grad()
        y_hat = net(x)
        l = loss_fn(y_hat,y)
        l.backward()
        optimizer.step()   
        
        batch_num = x.shape[0]
        total_loss += l.item()*batch_num
        total_samples += batch_num
        pred = torch.argmax(y_hat,dim=1)
        correct += (pred==y).sum().item()
         






