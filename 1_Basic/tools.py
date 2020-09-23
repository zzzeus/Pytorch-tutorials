from easydict import EasyDict as edict

d = edict({'foo':3, 'bar':{'x':1, 'y':2}})

print(dir(d))
