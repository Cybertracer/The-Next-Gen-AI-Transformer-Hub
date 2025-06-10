# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_bit_attention.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import BitAttention
    # BitLinear is needed to check isinstance for the projection layers
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import BitLinear
except ImportError as e:
    print(f"Error importing modules for testing in test_bit_attention.py: {e}")
    raise

class TestBitAttention(unittest.TestCase):

    def test_battention_instantiation(self):
        attn = BitAttention(dim=64, heads=4, dim_head=16)
        self.assertIsInstance(attn, nn.Module)
        self.assertEqual(attn.heads, 4)
        self.assertEqual(attn.dim_head, 16)
        self.assertIsInstance(attn.to_q, BitLinear)
        self.assertIsInstance(attn.to_k, BitLinear)
        self.assertIsInstance(attn.to_v, BitLinear)
        self.assertIsInstance(attn.to_out, BitLinear)

    def test_battention_forward_shape(self):
        dim = 128
        heads = 8
        dim_head = dim // heads
        attn = BitAttention(dim=dim, heads=heads, dim_head=dim_head)

        x = torch.randn(4, 10, dim) # Batch, Seq, Dim
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_battention_forward_with_mask(self):
        dim = 32
        heads = 4
        dim_head = dim // heads
        seq_len = 8
        attn = BitAttention(dim=dim, heads=heads, dim_head=dim_head)
        attn.eval()

        x = torch.randn(2, seq_len, dim)

        mask = torch.ones(2, seq_len).bool() # Mask is True for valid tokens
        if seq_len > 2: mask[0, -2:] = False # Last 2 tokens are padding for item 0
        if seq_len > 3: mask[1, -3:] = False # Last 3 tokens are padding for item 1

        out_masked = attn(x, mask=mask)
        self.assertEqual(out_masked.shape, x.shape)
        # Note: A more thorough test would involve inspecting attention weights.
        # This can be done by either returning them from forward or using hooks.
        # For example, if forward returned attn_weights:
        #   _, attn_weights_masked = attn(x, mask=mask, return_attn_weights=True)
        #   # Check that masked positions in attn_weights_masked are zero or near zero.
        #   # For head 0, batch 0:
        #   # For padding mask[0,j] = False, attn_weights_masked[0,0,:,j] should be all zeros.

    def test_battention_quantization_params_passed_to_bitlinear(self):
        dim = 16
        heads = 2
        dim_head = dim // heads

        custom_act_bits = 4
        custom_group_size = 16

        attn = BitAttention(dim=dim, heads=heads, dim_head=dim_head,
                            activation_bits=custom_act_bits,
                            weight_group_size=custom_group_size)

        # Check one of the BitLinear layers (e.g., to_q)
        self.assertEqual(attn.to_q.activation_bits, custom_act_bits)
        self.assertEqual(attn.to_q.group_size, custom_group_size)
        # Could check others too (to_k, to_v, to_out) but one is usually representative
        # if they are all instantiated similarly.
        self.assertEqual(attn.to_out.activation_bits, custom_act_bits)
        self.assertEqual(attn.to_out.group_size, custom_group_size)


    def test_battention_head_dim_consistency(self):
        attn = BitAttention(dim=64, heads=4, dim_head=17) # inner_dim = 17*4 = 68
        self.assertEqual(attn.inner_dim, 68)
        self.assertEqual(attn.to_q.out_features, 68) # Projects to inner_dim
        self.assertEqual(attn.to_out.in_features, 68) # Projects from inner_dim

        x = torch.randn(2, 5, 64) # Input dim is 64
        out = attn(x)
        self.assertEqual(out.shape, (2,5,64)) # Output dim should match input dim (64)


if __name__ == '__main__':
    unittest.main()
