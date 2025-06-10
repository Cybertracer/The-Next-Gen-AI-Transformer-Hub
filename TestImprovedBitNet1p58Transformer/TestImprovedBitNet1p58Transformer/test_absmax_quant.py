# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_absmax_quant.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
# This assumes tests are run from the repository root, or PYTHONPATH is set.
# For local 'python -m unittest discover' from root, this might not be strictly necessary
# if TestImprovedBitNet158Transformer is treated as a top-level package.
# However, to run this file directly for debugging, path adjustments are often needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import (
        absmax_quant,
        quantize_activations
        # BitLinear is not directly tested here but its context is for quantize_activations
    )
    # For testing quantize_activations context, we need a class that can be used with isinstance
    # We can use the actual BitLinear if it's simple enough not to pull too many dependencies for a unit test,
    # or mock it. The provided test uses a local MockBitLinear.
    # If we want to use the actual BitLinear, we'd import it:
    # from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import BitLinear

except ImportError as e:
    print(f"Error importing modules for testing in test_absmax_quant.py: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

# Mock BitLinear layer for testing quantize_activations
# This avoids needing the full BitLinear's dependencies if they are heavy for this unit test.
class MockBitLinear(nn.Module):
    def __init__(self, activation_bits_val, stochastic_rounding_val):
        super().__init__()
        self.activation_bits = activation_bits_val
        self.stochastic_rounding = stochastic_rounding_val
        # self.training is a built-in nn.Module attribute, True by default

class TestAbsmaxQuant(unittest.TestCase):

    def test_absmax_quant_basic(self):
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        dequant_x = absmax_quant(x, bits=8, stochastic=False)
        self.assertEqual(x.shape, dequant_x.shape)
        self.assertTrue(torch.isclose(dequant_x.max(), x.max(), atol=0.1))

        x_simple = torch.tensor([0.0, 63.5, 127.0])
        # Scale for x_simple: 127.0 / 127 = 1.0
        # q_x should be [0, round(63.5/1), round(127/1)] = [0, 64, 127]
        # dequant_simple = q_x * 1.0
        expected_simple_dequant = torch.tensor([0.0, 64.0, 127.0])
        dequant_simple = absmax_quant(x_simple, bits=8, stochastic=False)
        self.assertTrue(torch.allclose(dequant_simple, expected_simple_dequant, atol=1e-6))


    def test_absmax_quant_bits(self):
        x = torch.randn(10, 20) * 10
        dequant_x_4bit = absmax_quant(x, bits=4, stochastic=False)
        self.assertEqual(x.shape, dequant_x_4bit.shape)

        diff_8bit = (absmax_quant(x, bits=8) - x).abs().mean()
        diff_4bit = (dequant_x_4bit - x).abs().mean()
        # With fewer bits, quantization error should be larger.
        # It's possible for a specific random tensor that this doesn't hold if values fall luckily.
        # However, statistically, it should be greater.
        # We use a factor of 0.9 to allow for some edge cases but catch gross errors.
        if not torch.allclose(diff_4bit, diff_8bit) and diff_8bit > 1e-9: # Avoid division by zero or near-zero if x is tiny
             self.assertGreater(diff_4bit.item(), diff_8bit.item() * 0.9)


    def test_absmax_quant_stochastic(self):
        x = torch.arange(-10, 10, 0.1).float()

        # Deterministic (is_training=False)
        dequant_det1 = absmax_quant(x, bits=8, stochastic=True, is_training=False)
        dequant_det2 = absmax_quant(x, bits=8, stochastic=True, is_training=False)
        self.assertTrue(torch.allclose(dequant_det1, dequant_det2))

        # Deterministic (stochastic=False)
        dequant_det3 = absmax_quant(x, bits=8, stochastic=False, is_training=True) # is_training doesn't matter if stochastic=False
        dequant_det4 = absmax_quant(x, bits=8, stochastic=False, is_training=False)
        self.assertTrue(torch.allclose(dequant_det3, dequant_det4))
        self.assertTrue(torch.allclose(dequant_det1, dequant_det3)) # All deterministic paths should yield same result

        # Stochastic (training mode)
        results = [absmax_quant(x, bits=8, stochastic=True, is_training=True) for _ in range(20)] # Increased runs
        all_same = all(torch.allclose(results[0], res, atol=1e-7) for res in results[1:]) # Stricter atol for sameness check

        # It's theoretically possible all are same if noise never crosses a rounding boundary for any element.
        # For a tensor with diverse values like `x = torch.arange(-10, 10, 0.1)`, this is less likely.
        if x.numel() > 0: # Only assert if tensor is not empty
            self.assertFalse(all_same, "Stochastic rounding in training mode produced identical results over 20 runs, which is highly unlikely for this input.")


    def test_absmax_quant_zeros(self):
        x = torch.zeros(5, 5)
        dequant_x = absmax_quant(x, bits=8)
        self.assertTrue(torch.all(dequant_x == 0))

    def test_absmax_quant_positive_only(self):
        x = torch.rand(5, 5) * 10 + 1
        dequant_x = absmax_quant(x, bits=8)
        self.assertEqual(x.shape, dequant_x.shape)
        # Max value should be scaled to something close to original max after dequantization
        self.assertTrue(torch.isclose(dequant_x.max(), x.max(), atol=0.1)) # atol depends on Qb and scale


    def test_quantize_activations_helper(self):
        # We need to ensure that quantize_activations can import BitLinear from .quant_layers
        # For this unit test, we can mock it if the real one has complex dependencies.
        # The original solution used a local MockBitLinear, which is fine for unit testing `quantize_activations`.
        # We need to make sure that `isinstance(module, BitLinear)` in `quantize_activations`
        # correctly identifies our MockBitLinear as the type it's looking for.
        # This is best handled by temporarily patching where `quantize_activations` looks for `BitLinear`.

        original_bitlinear_in_quant_layers = None
        mock_quant_layers_module = None

        try:
            # Attempt to get the actual module where BitLinear is defined for patching
            # This assumes quantize_activations does "from .quant_layers import BitLinear"
            # and this test file is structured such that this relative import resolves.
            # The package for quantize_activations is ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant
            target_module_name = "ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant.quant_layers"

            if target_module_name in sys.modules:
                if hasattr(sys.modules[target_module_name], 'BitLinear'):
                    original_bitlinear_in_quant_layers = sys.modules[target_module_name].BitLinear
                sys.modules[target_module_name].BitLinear = MockBitLinear # Patch
            else:
                # If quant_layers hasn't been imported by anything yet, create a mock module
                mock_quant_layers_module = type(sys)('mock_quant_layers_for_test_absmax')
                mock_quant_layers_module.BitLinear = MockBitLinear
                sys.modules[target_module_name] = mock_quant_layers_module

            x = torch.randn(10, 20) * 5

            mock_layer_det_train = MockBitLinear(activation_bits_val=8, stochastic_rounding_val=False)
            mock_layer_det_train.train()
            out_det_train = quantize_activations(mock_layer_det_train, x)
            expected_det = absmax_quant(x, bits=8, stochastic=False)
            self.assertTrue(torch.allclose(out_det_train, expected_det))

            mock_layer_stoch_train = MockBitLinear(activation_bits_val=8, stochastic_rounding_val=True)
            mock_layer_stoch_train.train()
            out_stoch_train1 = quantize_activations(mock_layer_stoch_train, x)
            out_stoch_train2 = quantize_activations(mock_layer_stoch_train, x)
            self.assertFalse(torch.allclose(out_stoch_train1, expected_det, atol=1e-7), "Stochastic output (train) matched deterministic.")
            if x.numel() > 0:
                 self.assertFalse(torch.allclose(out_stoch_train1, out_stoch_train2, atol=1e-7), "Two stochastic outputs (train) were identical, unlikely.")

            mock_layer_stoch_eval = MockBitLinear(activation_bits_val=8, stochastic_rounding_val=True)
            mock_layer_stoch_eval.eval()
            out_stoch_eval = quantize_activations(mock_layer_stoch_eval, x)
            expected_stoch_eval = absmax_quant(x, bits=8, stochastic=True, is_training=False)
            self.assertTrue(torch.allclose(out_stoch_eval, expected_stoch_eval))
            self.assertTrue(torch.allclose(out_stoch_eval, expected_det)) # Should be same as fully deterministic

        finally:
            # Restore
            if target_module_name in sys.modules:
                if original_bitlinear_in_quant_layers is not None:
                    sys.modules[target_module_name].BitLinear = original_bitlinear_in_quant_layers
                elif mock_quant_layers_module is not None and sys.modules[target_module_name] == mock_quant_layers_module:
                    # If we inserted a brand new mock module, remove it
                    del sys.modules[target_module_name]


if __name__ == '__main__':
    unittest.main()
