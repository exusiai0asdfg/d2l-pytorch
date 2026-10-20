import torch
import random


def data_generate(w,b,data_num):
    x = torch.normal(0,1,(data_num,len(w)))
    y = x@w+b
    y+= torch.normal(0,0.1,(y.shape))
    y = y.reshape(-1,1)
    return(x,y)

def batch_data(x,y,batch_size):
    indice = list(range(len(x)))
    random.shuffle(indice)
    for i in range(0,len(indice),batch_size):              ####记住此处取indice的方式
        batch_indice = torch.tensor(indice[i:i+batch_size])
        x_batch = x[batch_indice]
        y_batch = y[batch_indice]
        yield x_batch,y_batch
        

### true 

true_w = torch.tensor([-4.3,2])
true_b = 2

### learn

def regloss(w,b,x,y):
    y_hat = x@w+b
    loss = (y_hat-y.reshape(y_hat.shape))**2*0.5  ###此处为了防止广播应该有.reshape
    return loss

def sgd(w,b,loss):
    loss.mean().backward()     #此处取mean()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
        w.grad.zero_()  ##记得清零可以使用.zero_()
        b.grad.zero_()
        return w,b
    



lr = 0.03

w = torch.tensor([0.,0.],requires_grad=True)
b = torch.tensor(0.,requires_grad=True)      #记得tensor需要标为浮点并开启grad


turn_times = 3

features,labels = data_generate(true_w,true_b,1000)

for turn in range(0,turn_times):
    for x,y in batch_data(features,labels,10):
        loss = regloss(w,b,x,y)
        w,b = sgd(w,b,loss)


print(f"true_w:{true_w},true_b:{true_b}")
print(f"learned_w:{w},learned_b:{b}")
