import numpy as np 

# Content
'''
1. create
zeros, ones

'''
## Create
a = np.zeros([2,2])
'''
[[0. 0.]
 [0. 0.]]
'''

a = np.ones([2,2])
'''
[[1. 1.]
 [1. 1.]]
'''

a = np.random.randint(0,6)
'''
5
'''

a = np.array([[2,3],[4,2]])
np.unique(a)
'''
[2 3 4]
'''

a = np.arange(9).reshape(3,3)
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''

## Operation
# a = np.random.randn(2,2)
# print(a)
# print(a.T)
# print(np.linalg.inv(a))

## Select
a = np.arange(9).reshape(3,3)
print(np.where(a>3))
'''
x,y = np.where(a>3)
(array([1, 1, 2, 2, 2], dtype=int64), array([1, 2, 0, 1, 2], dtype=int64))
'''
# print(np.where(a>3,True,False))
'''
[[False False False]
 [False  True  True]
 [ True  True  True]]
'''
