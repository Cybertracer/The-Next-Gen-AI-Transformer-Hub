# quant_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .absmean_quant import absmean_quant
from .absmax_quant import quantize_activations
from .quant_utils import QUANT_CONFIG

# TODO: Advanced - For specialized attention QKV projections, consider:
# 1. Per-channel quantization mode in absmean_quant if BitLinear's weight is [out, in]
#    and scaling is per 'out' channel. This could be a group_size modification or new mode.
# 2. A dedicated QKVParameter class or wrapper if Q, K, V weights need highly distinct
#    quantization logic or storage formats not covered by a generic BitLinear.
# 3. If per-channel scaling is added to absmean_quant, BitLinear might need a new parameter
#    to select that scaling mode for its self.weight.

class BitLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation_bits: int = QUANT_CONFIG.get('activation_bits', 8),
                 group_size: int = QUANT_CONFIG.get('weight_group_size', 128),
                 ternary_method: str = 'round',
                 stochastic_rounding: bool = False
                 ):
        super().__init__(in_features, out_features, bias)
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.ternary_method = ternary_method
        self.stochastic_rounding = stochastic_rounding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant_after_quant = absmean_quant(self.weight, self.group_size, self.ternary_method)
        output_before_act_quant = F.linear(x, w_dequant_after_quant, self.bias)
        output_after_act_quant = quantize_activations(self, output_before_act_quant)
        return output_after_act_quant

class BitEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int | None = None,
                 max_norm: float | None = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: torch.Tensor | None = None,
                 activation_bits: int = QUANT_CONFIG.get('activation_bits', 8),
                 stochastic_rounding: bool = False
                 ):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.activation_bits = activation_bits
        self.stochastic_rounding = stochastic_rounding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb_output = super().forward(input)
        quantized_emb_output = quantize_activations(self, emb_output)
        return quantized_emb_output


if __name__ == '__main__':
    # Existing BitLinear example
    print("Running quant_layers.py example (BitLinear):")
    in_features_example = 64
    out_features_example = 128
    batch_size_example = 4
    bitlinear_layer_round = BitLinear(in_features_example, out_features_example, ternary_method='round')
    bitlinear_layer_threshold = BitLinear(in_features_example, out_features_example, ternary_method='threshold')
    input_tensor_example_linear = torch.randn(batch_size_example, in_features_example) * 5
    print(f"Input tensor shape (for BitLinear): {input_tensor_example_linear.shape}")
    output_round = bitlinear_layer_round(input_tensor_example_linear)
    print(f"Output tensor shape (BitLinear, round): {output_round.shape}")
    print(f"Sample output (BitLinear, round, first row, first 5 vals): {output_round[0, :5]}")
    output_threshold = bitlinear_layer_threshold(input_tensor_example_linear)
    print(f"Output tensor shape (BitLinear, threshold): {output_threshold.shape}")
    print(f"Sample output (BitLinear, threshold, first row, first 5 vals): {output_threshold[0, :5]}")

    # Existing BitEmbedding example
    print("\nRunning quant_layers.py example (BitEmbedding):")
    num_embeddings_example = 100
    embedding_dim_example = 64
    bit_embedding_layer = BitEmbedding(num_embeddings_example, embedding_dim_example)
    print(f"BitEmbedding layer: {bit_embedding_layer}")
    input_ids_example_embedding = torch.randint(0, num_embeddings_example, (batch_size_example, 10))
    print(f"Input IDs shape (for BitEmbedding): {input_ids_example_embedding.shape}")
    embedded_output = bit_embedding_layer(input_ids_example_embedding)
    print(f"Output tensor shape (for BitEmbedding): {embedded_output.shape}")
    print(f"Sample embedded output (first batch, first token, first 5 vals): {embedded_output[0,0,:5]}")
    bit_embedding_stochastic = BitEmbedding(num_embeddings_example, embedding_dim_example, stochastic_rounding=True)
    bit_embedding_stochastic.train()
    print("\nBitEmbedding with stochastic rounding (training mode):")
    embedded_stochastic_output = bit_embedding_stochastic(input_ids_example_embedding)
    print(f"Sample stochastic embedded output (first batch, first token, first 5 vals):\n{embedded_stochastic_output[0,0,:5]}")
    bit_embedding_stochastic.eval()
    embedded_deterministic_eval_output = bit_embedding_stochastic(input_ids_example_embedding)
    print(f"Same layer in eval mode (should be deterministic):\n{embedded_deterministic_eval_output[0,0,:5]}")
