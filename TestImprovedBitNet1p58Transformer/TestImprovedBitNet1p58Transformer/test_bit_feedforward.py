# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_bit_feedforward.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import BitFeedForward
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import BitLinear
except ImportError as e:
    print(f"Error importing modules for testing in test_bit_feedforward.py: {e}")
    raise

class TestBitFeedForward(unittest.TestCase):

    def test_bfeedforward_instantiation(self):
        ffn = BitFeedForward(dim=64, hidden_dim=128)
        self.assertIsInstance(ffn, nn.Module)
        self.assertIsInstance(ffn.net[0], BitLinear) # First layer (BitLinear)
        self.assertIsInstance(ffn.net[1], nn.GELU)   # Default activation
        self.assertIsInstance(ffn.net[2], nn.Dropout) # Dropout after activation
        self.assertIsInstance(ffn.net[3], BitLinear) # Second layer (BitLinear)
        self.assertIsInstance(ffn.net[4], nn.Dropout) # Dropout after second BitLinear

    def test_bfeedforward_forward_shape(self):
        dim = 128
        hidden_dim = dim * 4
        ffn = BitFeedForward(dim=dim, hidden_dim=hidden_dim)

        x = torch.randn(4, 10, dim) # Batch, Seq, Dim
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_bfeedforward_custom_activation(self):
        ffn_relu = BitFeedForward(dim=64, hidden_dim=128, ffn_activation=nn.ReLU())
        self.assertIsInstance(ffn_relu.net[1], nn.ReLU)

        x = torch.randn(2, 5, 64)
        out = ffn_relu(x) # Check if it runs
        self.assertEqual(out.shape, x.shape)

    def test_bfeedforward_quantization_params_passed_to_bitlinear(self):
        dim = 16
        hidden_dim = 32

        custom_act_bits = 4
        custom_group_size = 16

        ffn = BitFeedForward(dim=dim, hidden_dim=hidden_dim,
                             activation_bits=custom_act_bits,
                             weight_group_size=custom_group_size)

        # Check first BitLinear layer (index 0)
        self.assertEqual(ffn.net[0].activation_bits, custom_act_bits)
        self.assertEqual(ffn.net[0].group_size, custom_group_size)
        # Check second BitLinear layer (index 3)
        self.assertEqual(ffn.net[3].activation_bits, custom_act_bits)
        self.assertEqual(ffn.net[3].group_size, custom_group_size)

    def test_bfeedforward_dropout_effect(self):
        dim = 32
        hidden_dim = 64
        dropout_rate = 0.5
        ffn_dropout = BitFeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout_rate)

        # Using a tensor of ones makes it easier to see if dropout has an effect
        # (i.e., if any elements become zero or are scaled).
        x = torch.ones(10, 20, dim)

        ffn_dropout.train()
        out_train1 = ffn_dropout(x)
        out_train2 = ffn_dropout(x)

        # With dropout, outputs of two consecutive forward passes on the same input should differ.
        self.assertFalse(torch.allclose(out_train1, out_train2, atol=1e-7),
                         "Outputs with dropout in train mode should differ across calls.")

        # Check if some elements are actually zeroed out or scaled by (1/(1-p))
        # This check is probabilistic. A simpler check is that the output is not identical to input.
        # Or that the sum of elements has changed due to scaling/zeroing.
        # If dropout_rate is high, many elements might be zero.
        # If dropout_rate is low, fewer elements are zero, but others are scaled up.
        # A robust check for dropout is that output is not equal to input when dropout > 0.
        if dropout_rate > 0:
            # Create a version without dropout for comparison
            ffn_no_dropout = BitFeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.0)
            # Ensure weights are the same for a fair comparison of dropout effect itself
            ffn_no_dropout.net[0].weight.data = ffn_dropout.net[0].weight.data.clone()
            if ffn_dropout.net[0].bias is not None:
                 ffn_no_dropout.net[0].bias.data = ffn_dropout.net[0].bias.data.clone()
            ffn_no_dropout.net[3].weight.data = ffn_dropout.net[3].weight.data.clone()
            if ffn_dropout.net[3].bias is not None:
                ffn_no_dropout.net[3].bias.data = ffn_dropout.net[3].bias.data.clone()

            ffn_no_dropout.eval() # No dropout in eval
            out_no_dropout = ffn_no_dropout(x)

            # Training output with dropout should be different from an output without dropout
            self.assertFalse(torch.allclose(out_train1, out_no_dropout, atol=1e-7),
                             "Training output with dropout should differ from output without dropout.")

        ffn_dropout.eval()
        out_eval1 = ffn_dropout(x)
        out_eval2 = ffn_dropout(x)
        self.assertTrue(torch.allclose(out_eval1, out_eval2, atol=1e-7),
                        "Outputs in eval mode should be deterministic (dropout off).")

        # Output in eval mode (dropout off) should be different from output in train mode (dropout on)
        # if dropout_rate > 0.
        if dropout_rate > 0:
            self.assertFalse(torch.allclose(out_train1, out_eval1, atol=1e-7),
                             "Training output (dropout on) and Eval output (dropout off) should differ if dropout_rate > 0.")


if __name__ == '__main__':
    unittest.main()
