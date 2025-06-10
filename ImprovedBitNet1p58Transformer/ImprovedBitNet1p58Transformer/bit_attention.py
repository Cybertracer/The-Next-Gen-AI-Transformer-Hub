# bit_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .bitnet_quant import BitLinear

class BitAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.0,
                 activation_bits: int = 8,
                 weight_group_size: int = 128):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # TODO: Advanced - Consider per-channel weight scaling for QKV projections if beneficial.
        # This might involve:
        # 1. Modifying BitLinear or absmean_quant to support per-channel scaling mode.
        #    (e.g., group_size=1 if weights are [out_features, in_features] and scaling is per out_feature).
        # 2. Potentially using a custom nn.Parameter wrapper (e.g., QKVParameter) if storage or specific
        #    quantization logic for Q, K, V weights independently is required before they are passed
        #    to a more generic linear layer.
        # For now, using BitLinear with its existing group-wise/per-tensor absmean quantization.

        self.to_q = BitLinear(dim, self.inner_dim, bias=False,
                              activation_bits=activation_bits, group_size=weight_group_size)
        self.to_k = BitLinear(dim, self.inner_dim, bias=False,
                              activation_bits=activation_bits, group_size=weight_group_size)
        self.to_v = BitLinear(dim, self.inner_dim, bias=False,
                              activation_bits=activation_bits, group_size=weight_group_size)

        self.to_out = BitLinear(self.inner_dim, dim,
                                activation_bits=activation_bits, group_size=weight_group_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            dots = dots.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(dots, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.to_out(out)

if __name__ == '__main__':
    # ... (existing __main__ block remains the same) ...
    print("Running bit_attention.py (BitAttention) example:")
    batch_size_example = 2
    seq_len_example = 16
    dim_example = 128
    heads_example = 8
    dim_head_example = dim_example // heads_example
    bit_attention_layer = BitAttention(dim=dim_example, heads=heads_example, dim_head=dim_head_example)
    print(f"BitAttention layer: {bit_attention_layer}")
    input_tensor_example = torch.randn(batch_size_example, seq_len_example, dim_example)
    print(f"Input tensor shape: {input_tensor_example.shape}")
    output_tensor_example = bit_attention_layer(input_tensor_example)
    print(f"Output tensor shape: {output_tensor_example.shape}")
    print("\nExample with padding mask:")
    padding_mask = torch.ones(batch_size_example, seq_len_example)
    padding_mask[:, -seq_len_example//2:] = 0
    output_masked_example = bit_attention_layer(input_tensor_example, mask=padding_mask)
    print(f"Output tensor shape (with padding mask): {output_masked_example.shape}")
