import torch
from torch import Tensor
from torch.nn import Module

def to_accelerator(self):
    if torch.cuda.is_available(): return self.cuda()
    try:
        if torch.backends.mps.is_available(): return self.to('mps')
    except:
        print("MPS backend not available")
    return self

Module.to_accelerator = to_accelerator
Tensor.to_accelerator = to_accelerator
