import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 3
output_size = 10 
lr =0.001
batch_size = 4
num_epochs = 1

# Transform
transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linears = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,16 * 5 *5)
        x = self.linears(x)
        return x



def imshow(imgs):
    imgs = imgs/2 + 0.5
    npimg = imgs.numpy()
    plt.imshow(np.transpose(npimg,[1,2,0]))
    plt.show()


# Train
def train():
    net = Net()
    net.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(),lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,(data,labels) in enumerate(trainloader,0):
            data = data.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            result = net(data)
            loss = criterion(result,labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(),'CNN.pth')

def predict():
    net = Net()
    net.load_state_dict(torch.load('CNN.pth'))
    dataiter = iter(testloader)
    images,labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ',' '.join('%5s'%classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

if __name__ == "__main__":
    # train()
    predict()