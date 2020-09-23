import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
nz = 100
ngf = 64
ndf = 64
nc = 3
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data,0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0,0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1), # torch.Size([4, 3, 64, 64])
            nn.Tanh()

        )
    def forward(self,x):
        return self.main(x)

noise = torch.randn([4,100,1,1])
g = Generator()
result = g(noise)
print(result.shape)

class  Discrimatior(nn.Module):
    def __init__(self):
        super(Discrimatior,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2, ndf*4,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4, ndf*8,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*8,1,4,2,0), # torch.Size([4, 1, 1, 1])
            nn.Sigmoid()

        )
    def forward(self,x):
        return self.main(x)

imgs = torch.randn([4,3,64,64])
d = Discrimatior()
result = d(imgs)
print(result.shape)

