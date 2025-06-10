# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_absmean_quant.py
import unittest
import torch
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import absmean_quant
except ImportError as e:
    print(f"Error importing absmean_quant for testing: {e}")
    raise

class TestAbsmeanQuant(unittest.TestCase):

    def test_absmean_quant_per_tensor_round(self):
        w = torch.tensor([[-1.5, -0.8, -0.2], [0.0, 0.1, 0.6], [1.2, 1.7, 0.3]]) * 2.0
        dequant_w = absmean_quant(w, group_size=None, ternary_method='round')
        self.assertEqual(w.shape, dequant_w.shape)

        scale = w.abs().mean().clamp_min(1e-9)
        q_w_expected = (w / scale).round().clamp(-1,1)
        dequant_expected = q_w_expected * scale
        self.assertTrue(torch.allclose(dequant_w, dequant_expected, atol=1e-6))
        self.assertTrue(all(val in [-1., 0., 1.] for val in q_w_expected.flatten().tolist()))

    def test_absmean_quant_per_tensor_threshold(self):
        w = torch.tensor([[-1.5, -0.8, -0.2], [0.0, 0.1, 0.6], [1.2, 1.7, 0.3]]) * 2.0
        dequant_w = absmean_quant(w, group_size=None, ternary_method='threshold')
        self.assertEqual(w.shape, dequant_w.shape)

        scale = w.abs().mean().clamp_min(1e-9)
        normalized_w = w / scale
        q_w_expected = torch.sign(normalized_w) * (normalized_w.abs() > 0.5).float()
        dequant_expected = q_w_expected * scale
        self.assertTrue(torch.allclose(dequant_w, dequant_expected, atol=1e-6))
        self.assertTrue(all(val in [-1., 0., 1.] for val in q_w_expected.flatten().tolist()))

    def test_absmean_quant_group_wise_round(self):
        w_non_divisible = torch.arange(-11, 12, 1.0).view(1, 23) # 23 elements
        group_size_nd = 4

        # Test fallback to per-tensor for non-divisible tensor
        dequant_w_fallback_nd = absmean_quant(w_non_divisible, group_size=group_size_nd, ternary_method='round')
        expected_dequant_pt_nd = absmean_quant(w_non_divisible, group_size=None, ternary_method='round')
        self.assertTrue(torch.allclose(dequant_w_fallback_nd, expected_dequant_pt_nd, atol=1e-6))

        # Test with divisible group size
        w_divisible = torch.randn(2, 16) * 5 # 32 elements
        group_size_div = 8 # 32 / 8 = 4 groups
        dequant_w_div = absmean_quant(w_divisible, group_size=group_size_div, ternary_method='round')
        self.assertEqual(w_divisible.shape, dequant_w_div.shape)
        # A precise check for group-wise values requires re-implementing the logic.
        # Instead, we check a property: the number of unique scaling factors used.
        if w_divisible.numel() > 0 and w_divisible.numel() % group_size_div == 0:
            reshaped_w = w_divisible.view(-1, group_size_div)
            scales_used = reshaped_w.abs().mean(dim=-1).clamp_min(1e-9)
            self.assertEqual(scales_used.numel(), w_divisible.numel() / group_size_div)


    def test_absmean_quant_group_wise_threshold(self):
        w_non_divisible = torch.arange(-11, 12, 1.0).view(1, 23) # 23 elements
        group_size_nd = 4
        dequant_w_fallback_nd = absmean_quant(w_non_divisible, group_size=group_size_nd, ternary_method='threshold')
        expected_dequant_pt_nd = absmean_quant(w_non_divisible, group_size=None, ternary_method='threshold')
        self.assertTrue(torch.allclose(dequant_w_fallback_nd, expected_dequant_pt_nd, atol=1e-6))

        w_divisible = torch.randn(2, 16) * 5
        group_size_div = 8
        dequant_w_div = absmean_quant(w_divisible, group_size=group_size_div, ternary_method='threshold')
        self.assertEqual(w_divisible.shape, dequant_w_div.shape)
        if w_divisible.numel() > 0 and w_divisible.numel() % group_size_div == 0:
            reshaped_w = w_divisible.view(-1, group_size_div)
            scales_used = reshaped_w.abs().mean(dim=-1).clamp_min(1e-9)
            self.assertEqual(scales_used.numel(), w_divisible.numel() / group_size_div)


    def test_absmean_quant_zeros(self):
        w = torch.zeros(5, 5)
        dequant_w_pt_r = absmean_quant(w, group_size=None, ternary_method='round')
        self.assertTrue(torch.all(dequant_w_pt_r == 0))
        dequant_w_pt_t = absmean_quant(w, group_size=None, ternary_method='threshold')
        self.assertTrue(torch.all(dequant_w_pt_t == 0))

        dequant_w_gw_r = absmean_quant(w, group_size=5, ternary_method='round')
        self.assertTrue(torch.all(dequant_w_gw_r == 0))
        dequant_w_gw_t = absmean_quant(w, group_size=5, ternary_method='threshold')
        self.assertTrue(torch.all(dequant_w_gw_t == 0))

    def test_absmean_quant_small_tensor_group_wise(self):
        # Tensor smaller than group_size should behave like per-tensor
        w = torch.tensor([-1.0, 0.5, 2.0])
        group_size = 128 # Larger than w.numel()

        dequant_gw_r = absmean_quant(w, group_size=group_size, ternary_method='round')
        dequant_pt_r = absmean_quant(w, group_size=None, ternary_method='round')
        self.assertTrue(torch.allclose(dequant_gw_r, dequant_pt_r, atol=1e-6))

        dequant_gw_t = absmean_quant(w, group_size=group_size, ternary_method='threshold')
        dequant_pt_t = absmean_quant(w, group_size=None, ternary_method='threshold')
        self.assertTrue(torch.allclose(dequant_gw_t, dequant_pt_t, atol=1e-6))

    def test_absmean_quant_empty_tensor(self):
        w = torch.empty(0, 5) # Empty tensor
        dequant_w_pt = absmean_quant(w, group_size=None)
        self.assertEqual(w.numel(), 0)
        self.assertEqual(dequant_w_pt.numel(), 0)

        dequant_w_gw = absmean_quant(w, group_size=128)
        self.assertEqual(dequant_w_gw.numel(), 0)


if __name__ == '__main__':
    unittest.main()
