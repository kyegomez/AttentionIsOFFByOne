import torch
from softmax_one.softmax_one import softmax_one

x = torch.randn(5)
y = softmax_one(x, dim=0)

print(y)
print(y.shape)