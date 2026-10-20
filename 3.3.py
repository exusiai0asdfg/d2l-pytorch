import numpy as np
import torch
from torch.utils import data

### true
def generate_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return x,y.reshape(-1,1)

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels  = generate_data(true_w,true_b,1000)

def load_array(data_arays,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_arays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)


from torch import nn

net = nn.Sequential(nn.Linear(2,1))
with torch.no_grad():
    net[0].weight.normal_(0, 0.01)
    net[0].bias.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(features),labels)
    print(f"epoch{epoch+1},loss{l:f}")

with torch.no_grad():
    w = net[0].weight
    b = net[0].bias
print(f"w_true:{true_w}")
print(f"w:{w}")
print(f"b_true:{true_b}")
print(f"b:{b}")
