import torch
import torch.nn as nn

# Forward declaration for type hinting if strict type checking were applied before BitLinear/BitConv2d are defined.
# String literals for type hints handle forward references automatically.

def absmax_quant(x: torch.Tensor, bits: int = 8, stochastic: bool = False, is_training: bool = False) -> torch.Tensor:
    """Per-tensor AbsMax quantization for activations.
    Returns dequantized output to simulate quantization effect.
    Optionally applies stochastic rounding if 'stochastic' is True and 'is_training' is True.
    """
    Qb = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp_min(1e-6) / Qb

    quant_target = x / scale

    if stochastic and is_training: # Apply stochastic rounding only during training
        noise = torch.rand_like(quant_target) - 0.5
        quant_target = quant_target + noise

    q_x = quant_target.round().clamp(-Qb, Qb)
    return q_x * scale  # Dequantized output

def quantize_activations(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply activation quantization based on module type.
    This function assumes that BitLinear and BitConv2d will be defined
    and imported from .quant_layers when this code is part of the larger package.
    It also assumes that the module might have 'activation_bits' and 'stochastic_rounding' attributes.
    """
    # Dynamically get BitLinear to avoid circular imports if this file is loaded before quant_layers
    # This is a common pattern.
    from .quant_layers import BitLinear #, BitConv2d # Assuming BitConv2d will also be in quant_layers

    # TODO: Add BitConv2d to the isinstance check once it's defined.
    if isinstance(module, BitLinear): # Or (BitLinear, BitConv2d)
        act_bits = getattr(module, 'activation_bits', 8) # Default to 8 if not present

        # Determine if stochastic rounding should be applied
        # Prefer module.stochastic_rounding if it exists
        stochastic_active = getattr(module, 'stochastic_rounding', False)
        is_training_mode = module.training # Standard nn.Module attribute (True if model.train(), False if model.eval())

        return absmax_quant(x, bits=act_bits, stochastic=stochastic_active, is_training=is_training_mode)
    return x

if __name__ == '__main__':
    print("Running absmax_quant.py example with stochastic rounding option:")

    test_tensor = torch.tensor([-2.7, -2.5, -2.3, -0.5, 0.0, 0.5, 2.3, 2.5, 2.7, 3.0]) * 2.0
    print(f"Original tensor: {test_tensor}")

    # Deterministic
    dequantized_deterministic = absmax_quant(test_tensor, bits=8, stochastic=False)
    print(f"Dequantized (deterministic): {dequantized_deterministic}")

    # Stochastic (simulating training mode)
    # Note: Stochastic results will vary per run due to random noise.
    print("\nSimulating stochastic rounding (training mode):")
    # Run a few times to see variability
    for i in range(3):
        dequantized_stochastic = absmax_quant(test_tensor, bits=8, stochastic=True, is_training=True)
        print(f"Run {i+1} Dequantized (stochastic): {dequantized_stochastic}")

    # Example for quantize_activations helper
    class DummyBitLinear(nn.Module):
        def __init__(self, activation_bits=8, stochastic_rounding_active=False):
            super().__init__()
            self.activation_bits = activation_bits
            self.stochastic_rounding = stochastic_rounding_active
            # self.training is a built-in attribute of nn.Module that is True by default

    print("\nTesting quantize_activations helper:")
    # Ensure .quant_layers can be mocked for the __main__ block if BitLinear is not yet fully usable
    # For this test, DummyBitLinear is sufficient.
    # To make `from .quant_layers import BitLinear` work in `quantize_activations` during this test,
    # we can add a mock to sys.modules if this file were run truly standalone before package setup.
    # However, since DummyBitLinear is local, and if BitLinear from .quant_layers is not found,
    # it would fail there. We need to ensure that quantize_activations can find a BitLinear.
    # For the purpose of this __main__ block, we can make DummyBitLinear the "BitLinear"
    # that quantize_activations will find, by a temporary sys.modules trick.

    import sys
    class MockQuantLayersModule:
        BitLinear = DummyBitLinear # Use our local DummyBitLinear as the one to be imported

    # Temporarily insert the mock module into sys.modules
    # The key should match how it's imported in quantize_activations: '.quant_layers' relative to this file's package
    # Assuming this file is in '.../bitnet_quant/'
    # The import path would be 'ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'
    # For a relative import '.quant_layers', Python resolves it within the current package.
    # If this __main__ is run, __package__ should be '...bitnet_quant'.
    # So, the effective full name for '.quant_layers' would be '...bitnet_quant.quant_layers'.

    # This gets a bit complex for __main__ blocks in submodules. A simpler way for testing
    # is to ensure the actual BitLinear in .quant_layers is robust enough or use a more direct test setup.
    # Given the current structure, we rely on the DummyBitLinear being sufficient for isinstance.
    # The from .quant_layers import BitLinear will attempt to import the actual one.
    # If that fails in a standalone __main__ run, this test might not work as intended without further mocking.
    # For now, let's assume that if .quant_layers.BitLinear exists, it's importable.
    # If not, the isinstance check might behave unexpectedly or fail.
    # The provided DummyBitLinear *is* a nn.Module, so it will pass isinstance(module, nn.Module)
    # but for isinstance(module, BitLinear) to work as intended in quantize_activations,
    # BitLinear must be the actual class from .quant_layers or this DummyBitLinear if mocked.

    # For the __main__ to robustly test quantize_activations, let's assume .quant_layers.BitLinear is importable
    # OR, we adjust quantize_activations to accept a type for BitLinear for testing, which is too much change.
    # The current DummyBitLinear is for the __main__ block only.
    # The `from .quant_layers import BitLinear` in `quantize_activations` will refer to the *actual* file.

    dummy_layer_deterministic = DummyBitLinear(activation_bits=8, stochastic_rounding_active=False)
    dummy_layer_stochastic_train = DummyBitLinear(activation_bits=8, stochastic_rounding_active=True)
    dummy_layer_stochastic_train.train() # Set to training mode

    dummy_layer_stochastic_eval = DummyBitLinear(activation_bits=8, stochastic_rounding_active=True)
    dummy_layer_stochastic_eval.eval() # Set to evaluation mode

    activations = torch.randn(2, 5) * 10
    print(f"Original activations for helper:\n{activations}")

    # To make this __main__ work without the actual .quant_layers.BitLinear being robustly importable
    # in all test environments, we can make the isinstance check in quantize_activations use
    # the DummyBitLinear for the test scenario. This is a common mocking pattern.
    # We can achieve this by temporarily replacing the BitLinear that quantize_activations imports.

    original_quant_layers_bitlinear = None
    if 'ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers' in sys.modules:
        if hasattr(sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'], 'BitLinear'):
            original_quant_layers_bitlinear = sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'].BitLinear
        sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'].BitLinear = DummyBitLinear
    else:
        # If quant_layers hasn't been imported yet by something else, we can create a mock module
        mock_ql_module = type(sys)('mock_quant_layers')
        mock_ql_module.BitLinear = DummyBitLinear
        sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'] = mock_ql_module


    output_deterministic = quantize_activations(dummy_layer_deterministic, activations)
    print(f"Helper output (deterministic, module training: {dummy_layer_deterministic.training}):\n{output_deterministic}")

    output_stochastic_train = quantize_activations(dummy_layer_stochastic_train, activations)
    print(f"Helper output (stochastic, module training: {dummy_layer_stochastic_train.training}):\n{output_stochastic_train}")

    output_stochastic_eval = quantize_activations(dummy_layer_stochastic_eval, activations)
    print(f"Helper output (stochastic active but module eval: {dummy_layer_stochastic_eval.training}):\n{output_stochastic_eval}")

    # Restore original BitLinear if it was patched, or remove the mock module
    if original_quant_layers_bitlinear:
        sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'].BitLinear = original_quant_layers_bitlinear
    elif 'ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers' in sys.modules and \
         sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers'].__name__ == 'mock_quant_layers':
        del sys.modules['ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers']
