import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 

#### Content
'''
1. create

'''

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

torch.randn([2,2])

torch.randint(0,10,[2,2])

torch.ones([2,2])

torch.zeros([2,2], dtype=torch.long)

a = torch.zero_(torch.ones([2,2]))
print(a)

# Get value
a=(torch.Tensor([4]))
print(a)
print(a.item())


a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# show Sigmoid
# m = nn.Sigmoid()
# input = torch.randn(1000)
# output = m(input)
# plt.plot(input.numpy(),output.numpy(),'.')
# plt.show()

#Show ReLU
# m = nn.ReLU()
# input = torch.randn(1000)
# output = m(input)
# plt.plot(input.numpy(),output.numpy(),'.')
# plt.show()

#Show ELU
# m = nn.ELU()
# input = torch.randn(1000)
# output = m(input)
# plt.plot(input.numpy(),output.numpy(),'.')
# plt.show()

#Show LeakyReLU
# m = nn.LeakyReLU()
# input = torch.randn(1000)
# output = m(input)
# plt.plot(input.numpy(),output.numpy(),'.')
# plt.show()

# Show Tanh
m = nn.Tanh()
input = torch.randn(1000)
output = m(input)
plt.plot(input.numpy(),output.numpy(),'.')
plt.show()