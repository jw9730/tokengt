import torch.nn as nn

from .batch_modules import Apply, Add
from .attn import SelfAttn as soft_SelfAttn


class EncLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_ff, n_heads, dropout=0., drop_mu=0., return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.add = Add()
        self.ln = Apply(nn.LayerNorm(dim_in))
        self.attn = soft_SelfAttn(n_heads=n_heads, d_in=dim_in, d_out=dim_in, d_qk=dim_qk, d_v=dim_v)  # <-
        self.ffn = Apply(nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim_ff, dim_in)
        ))

    def forward(self, G):
        h = self.ln(G)
        attn_score, h = self.attn(h)
        G = self.add(G, h)
        h = self.ffn(G)
        return (attn_score, self.add(G, h)) if self.return_attn else self.add(G, h)


class Encoder(nn.Module):
    def __init__(self, n_layers: int, dim_in, dim_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, drop_input=0.,
                 dropout=0., drop_mu=0., last_layer_n_heads=16):
        super().__init__()
        assert last_layer_n_heads >= 16
        self.input = Apply(
            nn.Sequential(
                nn.Linear(dim_in, dim_hidden),
                nn.Dropout(drop_input, inplace=True)
            )
        )
        layers = []
        for i in range(n_layers):
            layers.append(EncLayer(dim_hidden, dim_qk, dim_v, dim_ff, n_heads, dropout, drop_mu, return_attn=False))
        layers.append(
            EncLayer(dim_hidden, dim_qk, dim_v, dim_ff, last_layer_n_heads, dropout, drop_mu, return_attn=True))
        self.layers = nn.Sequential(*layers)

        self.output = Apply(
            nn.Sequential(
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_out)
            )
        )

    def forward(self, G):  # G.values: [bsize, max(n+e), 2*dim_hidden]
        G = self.input(G)  # G.values: [bsize, max(n+e), dim_hidden]
        attn_score, G = self.layers(G)  # attn_score: [bsize, last_layer_n_heads, |E|, |E|]
        # G.values: [bsize, max(n+e), dim_hidden]
        return attn_score, self.output(G)  # attn_score : [bsize, last_layer_n_heads, |E|, |E|]   # self.output(G).values: [bsize, max(n+e), dim_out]
