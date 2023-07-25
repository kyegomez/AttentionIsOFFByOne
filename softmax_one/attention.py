import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from softmax_one.softmax_one import softmax_one

# QuietAttention
class QuietAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)

        #prepend zero vector
        zero_vector = torch.zeros(1, q.size(1), d_k).to(q.device)
        q = torch.cat((zero_vector, q), dim=0)
        k = torch.cat((zero_vector, k), dim=0)
        v = torch.cat((zero_vector, v), dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = F.pad(mask, (1, 0))
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = softmax_one(scores, dim=-1)
        return torch.matmul(p_attn, v), p_attn