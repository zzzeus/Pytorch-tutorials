import torch 
import torch.nn as nn 
import torchvision
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision.transforms as T

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 28 * 28
output_size = 10
hidden_size = 100
num_epoch = 10
batch_size = 4
lr = 0.001

# Transform
transfrom = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5,],[0.5,])
    ]
)

# Dataset
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transfrom,download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transfrom,download=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=True,num_workers=2)

labels = [0,1,2,3,4,5,6,7,8,9]
# for i in train_dataset:
#     print(i[0].shape,i[1])
#     break

# Model
def getNet():
    return nn.Sequential(
        nn.Linear(in_features=input_size,out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size,out_features=output_size)
    )

net = getNet()
net.to(device) 

# Criterion and Optim
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr)

# Train
def train():
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i,(inputs,targets) in enumerate(train_dataloader,0):
            inputs,targets = inputs.reshape(-1, 28*28).to(device),targets.to(device)
            optim.zero_grad()
            result = net(inputs)
            loss = criterion(result,targets)
            
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('Epoch [{}/{}], loss: {:.4f}'.format(epoch+1,num_epoch,running_loss / 2000))
                running_loss = 0.0
    torch.save(net.state_dict(),'Logisitic.pth')

def imshow(imgs):
    imgs = imgs/2 + 0.5
    npimg = imgs.numpy()
    plt.imshow(np.transpose(npimg,[1,2,0]))
    plt.show()

def predict():
    net = getNet()
    net.load_state_dict(torch.load('Logisitic.pth'))
    with torch.no_grad():
        dataiter = iter(test_dataloader)
        images, labels = dataiter.next()
        predicted = net(images.reshape(-1, 28*28))
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(4)))
        _, predicted = torch.max(predicted, 1)

        print('Predicted: ', ' '.join('%5s' % predicted[j].item()
                                    for j in range(4)))

if __name__ == "__main__":
    # train()

    predict()