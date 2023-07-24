import math
import torch
import torch.nn.functional as F

# Define the softmax1 function
def softmax1(x, dim=None, _stacklevel=3, dtype=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


# Implement the scaled dot product attention with
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = softmax1(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn