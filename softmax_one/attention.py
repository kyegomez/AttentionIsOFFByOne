import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from softmax_one.softmax_one import softmax_one

# QuietAttention
class QuietAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Droput(dropout)

    def forward(self, q, k, v, mask=None):
        #Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)

        #create a 0 vector with same size as one quer vector in Q
        zero_vector = torch.zeros(1, q.size(1), d_k).to(q.device)

        #prepend the zero vector to queries, keys, and values
        q = torch.cat((zero_vector, q), dim=0)
        k = torch.cat((zero_vector, k), dim=0)
        v = torch.cat((zero_vector, v), dim=0)

        #compute the dot product of Q and K and scale by sqrt(d_k) for more stable gradientws
        scores = torch.matmul(q, k.tranpose(-2, -1)) / math.sqrt(d_k)

        #if a mask is provided, apply it to scores, (after extending the mask to account for the added zero vector)
        if mask is not None:
            mask = F.pad(mask, (1, 0))
            scores = scores.masked_fill(mask == 0, -1e9)

        #compute attention distribution using the modified softmax function
        p_attn = softmax_one(scores, dim=-1)

        #apply dropout to the attention distribution
        p_attn = self.dropout(p_attn)

        #compute the final output by multiplying the attention distribution with the value matrix v
        return torch.matmul(p_attn, v), p_attn