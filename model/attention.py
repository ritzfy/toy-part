import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, block_size, dropout=0.0, qkv_bias=False, need_weights=False):
        super().__init__()

        self.block_size = block_size
        # create putorch MHa class object
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        if self.block_size >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.block_size, :self.block_size]

        # Returns of tuple of tensors attention output and weights
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        output = self.proj(attn_output)

        return output