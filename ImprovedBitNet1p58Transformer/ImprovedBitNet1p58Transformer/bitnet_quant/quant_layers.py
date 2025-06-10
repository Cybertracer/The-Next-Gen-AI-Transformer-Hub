import torch
import torch.nn as nn
import torch.nn.functional as F

from .absmean_quant import absmean_quant
from .absmax_quant import quantize_activations # Use the helper from absmax_quant.py

class BitLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation_bits: int = 8,
                 # weight_bits: float = 1.58, # Not directly used in this forward pass
                 group_size: int = 128):
        super().__init__(in_features, out_features, bias)
        self.activation_bits = activation_bits
        # self.weight_bits = weight_bits # Store if needed for other logic or info
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BitLinear.
        1. Quantizes weights using absmean_quant (simulates ternary quantization but returns dequantized).
        2. Performs linear operation using these dequantized quantized weights.
        3. Quantizes the output activations using quantize_activations helper (which uses absmax_quant).
        """
        # Quantize weights: absmean_quant returns dequantized weights
        # that simulate the effect of ternary quantization.
        w_dequant_after_quant = absmean_quant(self.weight, self.group_size)

        # Perform linear operation with the effectively quantized (but dequantized) weights
        output_before_act_quant = F.linear(x, w_dequant_after_quant, self.bias)

        # Quantize output activations
        # quantize_activations helper function will use self.activation_bits
        output_after_act_quant = quantize_activations(self, output_before_act_quant)

        return output_after_act_quant

if __name__ == '__main__':
    print("Running quant_layers.py example (BitLinear):")

    # Setup:
    in_features_example = 64
    out_features_example = 128
    batch_size_example = 4

    # Create a BitLinear layer instance
    # Default: 8-bit activation, group_size 128 for weights
    bitlinear_layer = BitLinear(in_features_example, out_features_example)

    # Create a sample input tensor
    input_tensor_example = torch.randn(batch_size_example, in_features_example) * 5
    print(f"Input tensor shape: {input_tensor_example.shape}")
    print(f"BitLinear layer weight shape: {bitlinear_layer.weight.shape}")

    # Forward pass
    output_tensor_example = bitlinear_layer(input_tensor_example)
    print(f"Output tensor shape: {output_tensor_example.shape}")

    # You can inspect the output to see it's full-precision,
    # but it has undergone simulated quantization effects.
    print(f"Sample input (first row, first 5 vals): {input_tensor_example[0, :5]}")
    print(f"Sample output (first row, first 5 vals): {output_tensor_example[0, :5]}")

    # Example with different activation bitwidth (e.g., 4-bit, just for illustration)
    # Note: absmax_quant in our current version is hardcoded for Qb = 2**(bits-1)-1,
    # which is typical for symmetric quantization like int8.
    # For very low bits like 4, one might need different Qb calculations if asymmetric etc.
    # but the function should still run.
    bitlinear_layer_4bit_act = BitLinear(in_features_example, out_features_example, activation_bits=4)
    output_4bit_act = bitlinear_layer_4bit_act(input_tensor_example)
    print(f"Sample output (4-bit act, first row, first 5 vals): {output_4bit_act[0, :5]}")

    # Example with per-tensor weight quantization
    bitlinear_layer_per_tensor_w = BitLinear(in_features_example, out_features_example, group_size=None)
    output_per_tensor_w = bitlinear_layer_per_tensor_w(input_tensor_example)
    print(f"Sample output (per-tensor weight quant, first row, first 5 vals): {output_per_tensor_w[0, :5]}")
