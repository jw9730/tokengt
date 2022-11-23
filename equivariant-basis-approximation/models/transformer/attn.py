import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .batch_struct.sparse import batch_like


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        # Q, K: (bsize, nheads, |E|, d_qk)
        # V: (bsize, n_heads, |E|, d_v)
        # mask: = (bsize, 1, 1, |E|)

        dim_qk = Q.size(-1)
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(dim_qk)  # tensor(bsize, nheads, |E|, |E|)
        attn_score = attn_score.masked_fill(~mask, -1e9)  # tensor(bsize, nheads, |E|, |E|)
        attn_score = F.softmax(attn_score, dim=-1)  # tensor(bsize, nheads, |E|, |E|)
        output = torch.matmul(attn_score, V)  # tensor(bsize, nheads, |E|, d_v)
        return attn_score, output  # attn_score: tensor(bsize, nheads, |E|, |E|), output: tensor(bsize, nheads, |E|, d_v)


class SelfAttn(nn.Module):
    def __init__(self, n_heads=15, d_in=64, d_out=64, d_qk=512, d_v=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_out = d_out
        self.d_qk = d_qk
        self.d_v = d_v
        self.scaled_dot_attention = ScaledDotProductAttention()
        self.fc1 = nn.Linear(d_in, 2 * n_heads * d_qk)
        self.fc_v = nn.Linear(d_in, n_heads * d_v)
        self.fc_out = nn.Linear(n_heads * d_v, d_out)

    def forward(self, G):  # G.values: [bsize, |E|, d_in)
        bsize, e, _ = G.values.shape
        h = self.fc1(G.values)  # Tensor(bsize, |E|, 2*n_heads*d_qk)
        Q = h[..., :self.n_heads * self.d_qk].view(bsize, e, self.n_heads, self.d_qk)  # (bsize, |E|, n_heads, d_qk)
        K = h[..., self.n_heads * self.d_qk:].view(bsize, e, self.n_heads, self.d_qk)  # (bsize, |E|, n_heads, d_qk)

        V = self.fc_v(G.values)  # (bsize, |E|, n_heads*d_v)  
        V = V.masked_fill(~G.mask.unsqueeze(-1), 0)
        V = V.view(bsize, e, self.n_heads, self.d_v)  # (bsize, |E|, n_heads, d_v)

        Q = Q.transpose(1, 2)  # (bsize, n_heads, |E|, d_qk)
        K = K.transpose(1, 2)  # (bsize, n_heads, |E|, d_qk)
        V = V.transpose(1, 2)  # (bsize, n_heads, |E|, d_v)

        G_mask = G.mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, |E|)
        attn_score, prod_attn = self.scaled_dot_attention(Q, K, V, mask=G_mask)  # prod_attn: tensor(bsize, n_heads, |E|, d_v); attn_score: tensor(bsize, nheads, |E|, |E|)

        prod_attn = prod_attn.transpose(1, 2).contiguous()  # tensor(bsize, |E|, n_heads, d_v)
        prod_attn = prod_attn.view(bsize, e, -1)  # tensor(bsize, |E|, n_heads * d_v)

        output = self.fc_out(prod_attn)  # tensor(bsize, |E|, d_out)
        return attn_score, batch_like(G, output, skip_masking=False)
