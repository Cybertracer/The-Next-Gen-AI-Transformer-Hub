import torch
import torch.nn as nn # Added for BitLinear/BitConv2d type hint if needed, though not strictly necessary for isinstance checks with forward refs

# Forward declaration for type hinting if strict type checking were applied before BitLinear/BitConv2d are defined.
# In Python, string literals for type hints handle forward references automatically.
# So, 'BitLinear' and 'BitConv2d' in isinstance will resolve at runtime.

# Placeholder classes for BitLinear and BitConv2d if needed for the file to be fully self-contained
# for static analysis before other files are created. Otherwise, these are not strictly needed
# as the isinstance check will use the actual classes once quant_layers.py is imported elsewhere.
# For simplicity here, we'll assume these types will be available at runtime.
# class BitLinear(nn.Module): pass
# class BitConv2d(nn.Module): pass

def absmax_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-tensor AbsMax quantization for activations.
    Returns dequantized output to simulate quantization effect.
    """
    Qb = 2 ** (bits - 1) - 1
    # Clamp scale to avoid issues with all-zero tensors leading to NaN/inf.
    # If x.abs().max() is 0, scale becomes 1e-6 / Qb, which is a very small number.
    # q_x will be 0. Result will be 0. Correct.
    scale = x.abs().max().clamp_min(1e-6) / Qb

    q_x = (x / scale).round().clamp(-Qb, Qb)
    return q_x * scale  # Dequantized output

def quantize_activations(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply activation quantization based on module type.
    This function assumes that BitLinear and BitConv2d will be defined
    and imported from .quant_layers when this code is part of the larger package.
    """
    # Dynamically get BitLinear and BitConv2d to avoid circular imports
    # or issues if this file is imported before quant_layers.
    # This is a common pattern for handling types that might not be defined yet at import time.
    from .quant_layers import BitLinear #, BitConv2d # Assuming BitConv2d will also be in quant_layers

    # TODO: Add BitConv2d to the isinstance check once it's defined.
    # For now, only BitLinear is used as per user's provided BitLinear class.
    if isinstance(module, BitLinear): # Or (BitLinear, BitConv2d)
        return absmax_quant(x, bits=module.activation_bits)
    return x

if __name__ == '__main__':
    print("Running absmax_quant.py example:")

    # Example for absmax_quant
    test_tensor = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * 2.5
    print(f"Original tensor: {test_tensor}")
    dequantized_output = absmax_quant(test_tensor, bits=8)
    print(f"Dequantized output after absmax_quant (8-bit): {dequantized_output}")

    # Example for quantize_activations
    # We need a dummy BitLinear for this example to run standalone
    class DummyBitLinear(nn.Module):
        def __init__(self, activation_bits=8):
            super().__init__()
            self.activation_bits = activation_bits

    dummy_layer_8bit = DummyBitLinear(activation_bits=8)
    dummy_layer_4bit = DummyBitLinear(activation_bits=4) # Example for different bit width

    activations = torch.randn(2, 5) * 10 # Sample activations
    print(f"\nOriginal activations for quantize_activations:\n{activations}")

    # Temporarily mock .quant_layers for the __main__ block to run
    import sys
    class MockQuantLayers:
        BitLinear = DummyBitLinear # Use DummyBitLinear for the mock

    sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'] = MockQuantLayers()

    quantized_for_8bit_layer = quantize_activations(dummy_layer_8bit, activations)
    print(f"Output from quantize_activations (for 8-bit layer):\n{quantized_for_8bit_layer}")

    quantized_for_4bit_layer = quantize_activations(dummy_layer_4bit, activations)
    print(f"Output from quantize_activations (for 4-bit layer):\n{quantized_for_4bit_layer}")

    # Clean up the mock
    del sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers']

    # Test with all zeros
    all_zeros = torch.zeros(2,4)
    print(f"\nOriginal all zeros: {all_zeros}")
    dequant_zeros = absmax_quant(all_zeros, bits=8)
    print(f"Dequantized all zeros: {dequant_zeros}")
