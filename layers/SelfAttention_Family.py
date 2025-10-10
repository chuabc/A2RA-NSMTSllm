import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange, repeat
import math


class MultiHeadAttentionModel(nn.Module):
    def __init__(self, query_key_embed_dim, value_embed_dim, num_heads):
        super(MultiHeadAttentionModel, self).__init__()
        self.query_proj = nn.Linear(query_key_embed_dim, query_key_embed_dim)
        self.key_proj = nn.Linear(query_key_embed_dim, query_key_embed_dim)
        self.value_proj = nn.Linear(value_embed_dim, value_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(query_key_embed_dim, num_heads)

    def forward(self, query, key, value):
        query = query.view(query.size(0), -1)
        key = key.view(key.size(0), -1)
        value = value.view(value.size(0), -1)

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

        attn_output, _ = self.multihead_attn(Q, K, V)
        attn_output = attn_output.squeeze(0)
        return attn_output


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)
