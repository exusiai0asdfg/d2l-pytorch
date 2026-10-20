import torch
from torch import nn
from handwrite_relu1 import data

train_iter,test_iter = data()

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10)
)

def init_weights(m):
    if type(m) ==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

batch_size,lr,num_epochs = 256,0.1,10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.prameters(),lr=lr)

