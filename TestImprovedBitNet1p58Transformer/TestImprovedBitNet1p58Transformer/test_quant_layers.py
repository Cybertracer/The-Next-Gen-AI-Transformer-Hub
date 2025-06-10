# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_quant_layers.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import (
        BitLinear,
        BitEmbedding,
        QUANT_CONFIG # For default params if needed in tests
    )
except ImportError as e:
    print(f"Error importing modules for testing in test_quant_layers.py: {e}")
    raise

class TestQuantLayers(unittest.TestCase):

    def test_bitlinear_instantiation(self):
        bl = BitLinear(10, 20)
        self.assertIsInstance(bl, nn.Linear) # Inherits from nn.Linear
        self.assertEqual(bl.in_features, 10)
        self.assertEqual(bl.out_features, 20)
        # Check defaults (assuming QUANT_CONFIG is accessible and has these keys)
        self.assertEqual(bl.activation_bits, QUANT_CONFIG.get('activation_bits', 8))
        self.assertEqual(bl.group_size, QUANT_CONFIG.get('weight_group_size', 128))
        self.assertEqual(bl.ternary_method, 'round') # Default from BitLinear's __init__
        self.assertFalse(bl.stochastic_rounding) # Default from BitLinear's __init__


        bl_custom = BitLinear(10, 20, activation_bits=4, group_size=32, ternary_method='threshold', stochastic_rounding=True)
        self.assertEqual(bl_custom.activation_bits, 4)
        self.assertEqual(bl_custom.group_size, 32)
        self.assertEqual(bl_custom.ternary_method, 'threshold')
        self.assertTrue(bl_custom.stochastic_rounding)


    def test_bitlinear_forward_shape(self):
        bl = BitLinear(16, 32)
        x = torch.randn(4, 10, 16) # Batch, Seq, Features
        out = bl(x)
        self.assertEqual(out.shape, (4, 10, 32))

    def test_bitlinear_quantization_effects(self):
        x = torch.randn(2, 8) * 5

        # Use fixed weights for more reproducible comparisons of quantization effects
        fixed_weights = torch.arange(-16, 16, 1.0).view(4,8).float() * 0.1

        bl_default = BitLinear(8, 4, bias=False)
        bl_default.weight.data = fixed_weights.clone()
        bl_default.eval() # Ensure deterministic for baseline
        out_default = bl_default(x)

        # Change group_size for weight quantization
        bl_gs = BitLinear(8, 4, bias=False, group_size=4)
        bl_gs.weight.data = fixed_weights.clone()
        bl_gs.eval()
        out_gs = bl_gs(x)
        # Due to the nature of quantization, small changes might not always yield different float outputs
        # if the inputs/weights fall into specific bins. This is an inexact test.
        if x.numel() > 0 and fixed_weights.numel() > 0:
             self.assertFalse(torch.allclose(out_default, out_gs, atol=1e-7),
                             "Changing group_size should ideally change output for this input/weight config.")

        # Change ternary_method for weight quantization
        bl_tm = BitLinear(8, 4, bias=False, ternary_method='threshold')
        bl_tm.weight.data = fixed_weights.clone()
        bl_tm.eval()
        out_tm = bl_tm(x)
        if x.numel() > 0 and fixed_weights.numel() > 0:
            # This assertion is highly dependent on the specific weight values and inputs.
            # For some configurations, 'round' and 'threshold' might produce the same ternary values.
            # Consider a more targeted weight set if this fails inconsistently.
            # For this test, we expect a difference with the chosen fixed_weights.
            self.assertFalse(torch.allclose(out_default, out_tm, atol=1e-7),
                             "Changing ternary_method did not change output for this input/weight config. This might be okay for some specific values but check if unexpected.")


        # Change activation_bits
        bl_ab = BitLinear(8, 4, bias=False, activation_bits=4)
        bl_ab.weight.data = fixed_weights.clone()
        bl_ab.eval()
        out_ab = bl_ab(x)
        if x.numel() > 0 and fixed_weights.numel() > 0:
            self.assertFalse(torch.allclose(out_default, out_ab, atol=1e-7),
                            "Changing activation_bits should change output.")

        # Test stochastic rounding for activations
        bl_stoch = BitLinear(8, 4, bias=False, stochastic_rounding=True)
        bl_stoch.weight.data = fixed_weights.clone()

        bl_stoch.train()
        out_stoch1 = bl_stoch(x)
        out_stoch2 = bl_stoch(x)
        if x.numel() > 0:
            self.assertFalse(torch.allclose(out_stoch1, out_stoch2, atol=1e-7),
                            "Stochastic rounding should produce different outputs on different calls in train mode.")

        bl_stoch.eval()
        out_stoch_eval1 = bl_stoch(x)
        out_stoch_eval2 = bl_stoch(x)
        self.assertTrue(torch.allclose(out_stoch_eval1, out_stoch_eval2, atol=1e-7),
                        "Stochastic rounding in eval mode should be deterministic.")
        # And it should be different from the training mode stochastic output (probabilistically)
        # And also different from the default (non-stochastic) output due to rounding differences
        self.assertFalse(torch.allclose(out_default, out_stoch_eval1, atol=1e-7),
                         "Deterministic output of stochastic_rounding=True layer (in eval) should differ from default stochastic_rounding=False layer if rounding choices differ.")


    def test_bitembedding_instantiation(self):
        be = BitEmbedding(100, 128)
        self.assertIsInstance(be, nn.Embedding)
        self.assertEqual(be.num_embeddings, 100)
        self.assertEqual(be.embedding_dim, 128)
        self.assertEqual(be.activation_bits, QUANT_CONFIG.get('activation_bits', 8))
        self.assertFalse(be.stochastic_rounding)

        be_custom = BitEmbedding(100, 128, activation_bits=4, stochastic_rounding=True)
        self.assertEqual(be_custom.activation_bits, 4)
        self.assertTrue(be_custom.stochastic_rounding)

    def test_bitembedding_forward_shape(self):
        be = BitEmbedding(50, 64)
        x_indices = torch.randint(0, 50, (4, 10)) # Batch, Seq
        out = be(x_indices)
        self.assertEqual(out.shape, (4, 10, 64))

    def test_bitembedding_quantization_effects(self):
        x_indices = torch.arange(0, 10).unsqueeze(0) # (1,10)

        # Use fixed weights for reproducibility
        fixed_emb_weights = torch.arange(0, 10*16).view(10,16).float() * 0.01

        be_8bit = BitEmbedding(10, 16, activation_bits=8, stochastic_rounding=False)
        be_8bit.weight.data = fixed_emb_weights.clone()
        be_8bit.eval()
        out_8bit = be_8bit(x_indices)

        be_4bit = BitEmbedding(10, 16, activation_bits=4, stochastic_rounding=False)
        be_4bit.weight.data = fixed_emb_weights.clone()
        be_4bit.eval()
        out_4bit = be_4bit(x_indices)
        if x_indices.numel() > 0 :
            self.assertFalse(torch.allclose(out_8bit, out_4bit, atol=1e-7),
                            "Different activation_bits in BitEmbedding should change output.")

        # Test stochastic rounding
        be_stoch = BitEmbedding(10, 16, activation_bits=8, stochastic_rounding=True)
        be_stoch.weight.data = fixed_emb_weights.clone()

        be_stoch.train()
        out_stoch1 = be_stoch(x_indices)
        out_stoch2 = be_stoch(x_indices)
        if x_indices.numel() > 0:
            self.assertFalse(torch.allclose(out_stoch1, out_stoch2, atol=1e-7),
                            "BitEmbedding stochastic rounding should produce different outputs in train mode.")

        be_stoch.eval()
        out_stoch_eval1 = be_stoch(x_indices)
        out_stoch_eval2 = be_stoch(x_indices)
        self.assertTrue(torch.allclose(out_stoch_eval1, out_stoch_eval2, atol=1e-7),
                        "BitEmbedding stochastic rounding in eval mode should be deterministic.")
        # Compare eval stochastic to original 8-bit deterministic
        # They should be different if stochastic rounding (even when deterministic in eval)
        # rounds differently than the default round() in some cases.
        if x_indices.numel() > 0:
            self.assertFalse(torch.allclose(out_8bit, out_stoch_eval1, atol=1e-7),
                             "BitEmbedding eval output with stochastic_rounding=True flag should differ from default if rounding choices differ.")


if __name__ == '__main__':
    unittest.main()
