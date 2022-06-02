import torch
import utils as ut
import numpy as np 
import time

a = torch.Tensor(128,32)
b = torch.Tensor(128,32)

x = torch.Tensor(128, 3, 32, 32)
device = "cpu"

t0 = time.time()
transforms = ut.get_transforms(0.5, 0.3)  

t1 = time.time()
print(t1-t0)

x_aug = transforms(x)

t2 = time.time()
print(t2-t1)

loss = ut.MMD(a, b, 'multiscale', device)

print('time passed {}'.format(time.time()-t2))


