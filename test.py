import numpy as np
import torch

def fun(c):
    c[0,0]=1
    return c
a=torch.zeros(1000,1000)
b=fun(a)
print(a[0,0])