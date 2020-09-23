import torch 
import torch.nn as nn
import numpy as np 
import random
import matplotlib.pyplot as plt

input_size,output_size  =1,1
lr = 0.001
num_epochs=30

# Dataset
x_train = [i for i in range(20)]
y_train = [i +random.randint(0,2) for i in x_train]

x_train=np.array(x_train,dtype=np.float32)
y_train=np.array(y_train,dtype=np.float32)
x_train,y_train = x_train[...,np.newaxis],y_train[...,np.newaxis]

# Model
net = nn.Linear(input_size,output_size)

# Criterion and Optim
criterion = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(),lr)

# Train
def train():
    for  epoch in range(num_epochs):
        inputs,targets = torch.from_numpy(x_train),torch.from_numpy(y_train)

        predicted = net(inputs)
        loss = criterion(predicted,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (epoch+1)%5==0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,num_epochs,loss.item()))
    torch.save(net.state_dict(),'linear.pth')

# Predict
def predict():
    inputs= torch.from_numpy(x_train)
    predicted = net(inputs).detach().numpy()
    plt.plot(x_train,y_train,'ro',label='Original data')
    plt.plot(x_train,predicted,label='Fitted line')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()

    predict()
