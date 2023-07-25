import math
import torch
import torch.nn.functional as F

# Define the softmax_one function with added one in the denominator , which helps to reduce
#the negative impact impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


# Implement the scaled dot product attention with GhostSoftmax
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = softmax_one(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn