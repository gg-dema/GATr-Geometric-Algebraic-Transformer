import numpy as np

import torch
import torch.nn as nn
from einops import rearrange
from src.GA_layers.Linears import EquiLinear
from src.Utils.geometric_utility import inner_product_through_reverse



class EquiMultiHeadAttention(nn.Module):

    def __init__(self, n_head, hidden_channels):
        super(EquiMultiHeadAttention, self).__init__()  
        self.n_head = n_head
        self.hidden_channels = hidden_channels
        self.qkv_layer = EquiLinear(in_channels=hidden_channels,
                                    out_channels=n_head * 3 * hidden_channels)
        self.out_proj = EquiLinear(in_channels=n_head * hidden_channels, out_channels=hidden_channels)


    def forward(self, x):
        
        # encode key vector value tensor
        qkv = self.qkv_layer(x)

        qkv = rearrange(
            qkv,
            'batch seq (heads hidden_c n_proj) mv_dim -> n_proj batch heads seq hidden_c mv_dim',
            heads=self.n_head, 
            n_proj=3, 
            hidden_c=self.hidden_channels
        )
        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        q_mv = rearrange(q, "... items_out channels x -> ... items_out 1 channels x")
        k_mv = rearrange(k, "... items_in channels x -> ... 1 items_in channels x")
        h = inner_product_through_reverse(q_mv, k_mv)  # (..., items_out, items_in, channels)
        
        # normalize
        attn_weights = torch.sum(h, dim=-1) / np.sqrt(8*4)

        # Softmax
        attn_weights = attn_weights.softmax(dim=-1)  # Softmax over items_in
        
        # calc attention score for each head
        attentions = torch.einsum(
                    "... j i, ... i c x -> ... j c x", attn_weights, v
                ).squeeze(0)

        # mix up channels and heads ---> using multi-head projection 
        attentions = rearrange(attentions, 
                        "... head items_out channels x -> ... items_out (channels head) x"
        )
        return self.out_proj(attentions)
        
attention = EquiMultiHeadAttention(n_head=4, hidden_channels=4)