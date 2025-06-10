import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming BitLinear is available from .bitnet_quant.quant_layers
from .bitnet_quant import BitLinear
# If QUANT_CONFIG is used to get activation_bits, import it too
# from .bitnet_quant import QUANT_CONFIG

class BitFeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0,
                 activation_bits: int = 8, # For output of BitLinear layers
                 weight_group_size: int = 128, # For BitLinear layers
                 ffn_activation: nn.Module = None): # Allow custom FFN activation
        """
        BitFeedForward module using BitLinear layers.
        Typically consists of: Linear -> Activation -> Dropout -> Linear -> Dropout

        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
            activation_bits (int): Bits for activation quantization in BitLinear layers.
            weight_group_size (int): Group size for weight quantization in BitLinear layers.
            ffn_activation (nn.Module, optional): Activation function to use. Defaults to nn.GELU().
        """
        super().__init__()

        if ffn_activation is None:
            ffn_activation = nn.GELU()

        self.net = nn.Sequential(
            BitLinear(dim, hidden_dim,
                      activation_bits=activation_bits, group_size=weight_group_size),
            ffn_activation,
            nn.Dropout(dropout), # Dropout after activation
            BitLinear(hidden_dim, dim,
                      activation_bits=activation_bits, group_size=weight_group_size),
            nn.Dropout(dropout) # Dropout after the second linear layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BitFeedForward.
        BitLinear layers internally handle their activation quantization.
        """
        return self.net(x)

if __name__ == '__main__':
    print("Running bit_feedforward.py (BitFeedForward) example:")

    # Parameters
    batch_size_example = 2
    seq_len_example = 16
    dim_example = 128
    hidden_dim_example = dim_example * 4 # Common practice: FFN hidden dim is 4x input/output dim
    dropout_example = 0.1

    # Create BitFeedForward layer
    # Using default quantization parameters from BitLinear (8-bit act, 128 group size for weights)
    # and default GELU activation
    bit_ffn_layer = BitFeedForward(dim=dim_example,
                                   hidden_dim=hidden_dim_example,
                                   dropout=dropout_example)
    print(f"BitFeedForward layer (with GELU): {bit_ffn_layer}")

    # Create a sample input tensor
    input_tensor_example = torch.randn(batch_size_example, seq_len_example, dim_example)
    print(f"Input tensor shape: {input_tensor_example.shape}")

    # Forward pass
    output_tensor_example = bit_ffn_layer(input_tensor_example)
    print(f"Output tensor shape: {output_tensor_example.shape}") # Expected: (batch, seq_len, dim)
    print(f"Sample output (first batch, first token, first 5 values): {output_tensor_example[0,0,:5]}")

    # Example with a different activation (ReLU)
    print("\nExample with ReLU activation:")
    bit_ffn_relu_layer = BitFeedForward(dim=dim_example,
                                        hidden_dim=hidden_dim_example,
                                        dropout=dropout_example,
                                        ffn_activation=nn.ReLU())
    print(f"BitFeedForward layer (with ReLU): {bit_ffn_relu_layer}")
    output_relu_example = bit_ffn_relu_layer(input_tensor_example)
    print(f"Output tensor shape (ReLU): {output_relu_example.shape}")
    print(f"Sample output (ReLU, first batch, first token, first 5 values): {output_relu_example[0,0,:5]}")
