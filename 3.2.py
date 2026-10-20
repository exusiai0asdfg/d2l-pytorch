import random
import torch

random.seed(42)
torch.manual_seed(42)

def generate_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = x @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return x,y.reshape(-1,1)



def data_iter(batch_size,features,labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0,num_example,batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_example)]
        )
        yield features[batch_indices], labels[batch_indices]
        
        
        
        
def linreg(x,w,b):
    return x @ w + b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad 
            param.grad.zero_()      
        
### true 
true_w = torch.tensor([[2.0], [-3.4]])
true_b = 4.2
features , labels= generate_data(true_w,true_b,1000)  
        
        
        
### model  
        
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


     

  
lr  = 0.03
batch_size = 10
num_epochs = 3




for epoch in range(num_epochs):
    for x_batch,y_batch in data_iter(batch_size,features,labels):
        loss = squared_loss(linreg(x_batch,w,b),y_batch)
        loss.mean().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_loss = squared_loss(linreg(features,w,b),labels)
    
    print(
        f"epoch{epoch +1}"
        f"loss{train_loss.mean().item():.6f}"
        
    )
    
print('learned w:',w)
print('true w:',true_w)
print('learned b:',b)
print('true b:',true_b)