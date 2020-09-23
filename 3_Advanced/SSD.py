import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
import numpy as np
from torchviz import make_dot, make_dot_from_trace

class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,3,1,0),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,3),

            nn.Conv2d(64,128,3,1,0),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,3),

            nn.Conv2d(128,256,3,1,0),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,3),

            nn.Conv2d(256,512,3,1,0),
            nn.ReLU(),
            nn.Conv2d(512,512,3,2,0),
            nn.ReLU(),
            nn.MaxPool2d(2,3),
        )

    def forward(self,x):
        return self.main(x)

ssd = SSD()
result = ssd(torch.randn([4,3,300,300]))
print(result.shape)

from torchvision.models import vgg16

model = vgg16(pretrained=True)

x = torch.randn(1,3,300,300)
make_dot(model(x), params=dict(model.named_parameters())).render('vgg.pdf', view=True)  

