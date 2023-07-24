import torch
from softmax_one import softmax1

x = torch.randn(5)
y = softmax1(x, dim=0)

