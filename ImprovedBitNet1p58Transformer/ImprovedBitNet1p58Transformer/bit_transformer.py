import torch
import torch.nn as nn

from .bit_attention import BitAttention
from .bit_feedforward import BitFeedForward
from .norm import RMSNorm
from .bitnet_quant import QUANT_CONFIG, BitEmbedding # BitEmbedding is now imported

class BitTransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 dropout: float = 0.1,
                 activation_bits: int = QUANT_CONFIG.get('activation_bits', 8),
                 weight_group_size: int = QUANT_CONFIG.get('weight_group_size', 128),
                 # BitLinear specific params that could be passed down if needed:
                 ternary_method: str = 'round',
                 stochastic_rounding_activations: bool = False
                ):
        super().__init__()
        # Note: BitAttention and BitFeedForward internally create BitLinear layers.
        # To control ternary_method or stochastic_rounding for those BitLinear layers,
        # BitAttention/BitFeedForward __init__ methods would need to accept these params
        # and pass them to their BitLinear instances.
        # For now, they use their own defaults or what's set in BitLinear's __init__.
        self.attention = BitAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                      activation_bits=activation_bits, weight_group_size=weight_group_size)
        self.norm1 = RMSNorm(dim)
        self.feed_forward = BitFeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout,
                                           activation_bits=activation_bits, weight_group_size=weight_group_size)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x


class BitTransformer(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 num_classes: int,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 # Quantization params for BitLinear layers in Attention/FFN
                 activation_bits: int = QUANT_CONFIG.get('activation_bits', 8),
                 weight_group_size: int = QUANT_CONFIG.get('weight_group_size', 128),
                 # New parameters for embedding quantization
                 quantize_embedding: bool = QUANT_CONFIG.get('quantize_embeddings', False),
                 embedding_activation_bits: int = QUANT_CONFIG.get('activation_bits', 8), # Default to general activation_bits
                 embedding_stochastic_rounding: bool = False
                ):
        super().__init__()

        self.quantize_embedding = quantize_embedding
        if self.quantize_embedding:
            self.embedding = BitEmbedding(
                num_embeddings=num_tokens,
                embedding_dim=dim,
                activation_bits=embedding_activation_bits,
                stochastic_rounding=embedding_stochastic_rounding
                # padding_idx could be passed here if needed
            )
        else:
            self.embedding = nn.Embedding(num_tokens, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Pass relevant quantization parameters to each block
            # Note: ternary_method and stochastic_rounding for BitLinear layers within
            # BitAttention/BitFeedForward are currently managed by those classes' defaults
            # or would need to be explicitly passed if more granular control is desired here.
            self.layers.append(BitTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                activation_bits=activation_bits,
                weight_group_size=weight_group_size
            ))

        self.final_norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes) # Standard Linear for logits


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len = x.shape

        x_emb = self.embedding(x)
        if x_emb.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum positional embedding length ({self.pos_embedding.shape[1]})")

        x_pos = self.pos_embedding[:, :seq_len]

        x = x_emb + x_pos

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.to_logits(x)
        return logits

if __name__ == '__main__':
    print("Running bit_transformer.py with optional BitEmbedding example:")

    test_num_tokens = 1000
    test_dim = 64
    test_depth = 1
    test_heads = 4
    test_dim_head = test_dim // test_heads
    test_mlp_dim = test_dim * 4
    test_num_classes = 10
    test_max_seq_len = 32

    print("\n--- Model with Standard Embedding ---")
    model_std_emb = BitTransformer(
        num_tokens=test_num_tokens, dim=test_dim, depth=test_depth, heads=test_heads,
        dim_head=test_dim_head, mlp_dim=test_mlp_dim, num_classes=test_num_classes,
        max_seq_len=test_max_seq_len, quantize_embedding=False
    )
    print(f"Embedding type: {type(model_std_emb.embedding)}")
    dummy_input_tokens = torch.randint(0, test_num_tokens, (2, 10))
    output_std_emb = model_std_emb(dummy_input_tokens)
    print(f"Output shape (std_emb): {output_std_emb.shape}")

    print("\n--- Model with BitEmbedding (8-bit, deterministic, eval mode) ---")
    model_bit_emb_eval = BitTransformer(
        num_tokens=test_num_tokens, dim=test_dim, depth=test_depth, heads=test_heads,
        dim_head=test_dim_head, mlp_dim=test_mlp_dim, num_classes=test_num_classes,
        max_seq_len=test_max_seq_len,
        quantize_embedding=True,
        embedding_activation_bits=8,
        embedding_stochastic_rounding=False # Explicitly False for this test
    )
    model_bit_emb_eval.eval() # Ensure deterministic for BitEmbedding
    print(f"Embedding type: {type(model_bit_emb_eval.embedding)}")
    output_bit_emb_eval = model_bit_emb_eval(dummy_input_tokens)
    print(f"Output shape (bit_emb eval): {output_bit_emb_eval.shape}")
    print(f"Sample output (bit_emb eval, first val): {output_bit_emb_eval[0,0,0]}")

    print("\n--- Model with BitEmbedding (8-bit, stochastic_rounding=True, but in eval mode) ---")
    model_bit_emb_stoch_flag_eval_mode = BitTransformer(
        num_tokens=test_num_tokens, dim=test_dim, depth=test_depth, heads=test_heads,
        dim_head=test_dim_head, mlp_dim=test_mlp_dim, num_classes=test_num_classes,
        max_seq_len=test_max_seq_len,
        quantize_embedding=True,
        embedding_activation_bits=8,
        embedding_stochastic_rounding=True # Stochastic flag is True
    )
    model_bit_emb_stoch_flag_eval_mode.eval() # BUT model is in eval mode
    print(f"Embedding type: {type(model_bit_emb_stoch_flag_eval_mode.embedding)}")
    output_bit_emb_stoch_flag_eval_mode = model_bit_emb_stoch_flag_eval_mode(dummy_input_tokens)
    print(f"Output shape (bit_emb stoch_flag but eval_mode): {output_bit_emb_stoch_flag_eval_mode.shape}")
    print(f"Sample output (bit_emb stoch_flag but eval_mode, first val): {output_bit_emb_stoch_flag_eval_mode[0,0,0]}")


    print("\n--- Model with BitEmbedding (8-bit, stochastic_rounding=True, and in train mode) ---")
    model_bit_emb_stochastic_train_mode = BitTransformer(
        num_tokens=test_num_tokens, dim=test_dim, depth=test_depth, heads=test_heads,
        dim_head=test_dim_head, mlp_dim=test_mlp_dim, num_classes=test_num_classes,
        max_seq_len=test_max_seq_len,
        quantize_embedding=True,
        embedding_activation_bits=8,
        embedding_stochastic_rounding=True # Stochastic flag is True
    )
    model_bit_emb_stochastic_train_mode.train() # AND model is in train mode
    print(f"Embedding type: {type(model_bit_emb_stochastic_train_mode.embedding)}")
    output_bit_emb_stoch_train_mode = model_bit_emb_stochastic_train_mode(dummy_input_tokens)
    print(f"Output shape (bit_emb stochastic train_mode): {output_bit_emb_stoch_train_mode.shape}")
    print(f"Sample output (bit_emb stochastic train_mode, first val): {output_bit_emb_stoch_train_mode[0,0,0]}")
    # Another run to see if it's different due to stochasticity
    output_bit_emb_stoch_train_mode_run2 = model_bit_emb_stochastic_train_mode(dummy_input_tokens)
    print(f"Sample output (bit_emb stochastic train_mode, first val, run 2): {output_bit_emb_stoch_train_mode_run2[0,0,0]}")
    # Check if they are different (they should be with high probability if stochastic rounding is working)
    if not torch.allclose(output_bit_emb_stoch_train_mode, output_bit_emb_stoch_train_mode_run2):
        print("Outputs from two runs with stochastic rounding in train mode are different, as expected.")
    else:
        print("Warning: Outputs from two runs with stochastic rounding in train mode are the same. Check implementation.")
