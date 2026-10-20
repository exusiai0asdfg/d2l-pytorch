import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data():
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
    return train_iter,test_iter


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(a,x)

def net(x):
    x = x.reshape(-1,num_inputs)
    h = relu(x@w1+b1)
    o = h@w2+b2
    return o


if __name__ == "__main__":
    train_iter,test_iter = data()

    num_inputs = 784
    num_hiddens = 256
    num_outputs = 10
    w1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    w2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [w1,b1,w2,b2]


    lr = 0.1
    num_epochs = 10


    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params, lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        correct = 0

        for x, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(x)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()

            batch_n = x.shape[0]
            total_loss += l.item()*batch_n
            total_samples += batch_n
            pred = torch.argmax(y_hat,dim = 1)
            correct += (pred == y).sum().item()
        
        train_acc = correct / total_samples
        avg_loss = total_loss/total_samples
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, train_acc = {train_acc:.2%}")





    test_correct = 0
    test_sample = 0

    with torch.no_grad():
        for x,y in test_iter:
            y_hat = net(x)
            pred = torch.argmax(y_hat,dim=1)
            test_correct += (y==pred).sum().item()
            test_sample += len(x)
            
        test_acc = test_correct/test_sample
        print(f"Test accuracy: {test_acc:.2%}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
