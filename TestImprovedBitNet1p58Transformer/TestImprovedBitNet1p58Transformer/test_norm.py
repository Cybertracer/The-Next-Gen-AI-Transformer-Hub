# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_norm.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.norm import RMSNorm
except ImportError as e:
    print(f"Error importing RMSNorm for testing: {e}")
    raise

class TestRMSNorm(unittest.TestCase):

    def test_rmsnorm_instantiation(self):
        norm = RMSNorm(dim=64)
        self.assertIsInstance(norm, nn.Module)
        self.assertEqual(norm.weight.shape, (64,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(64))) # Default weight is ones

    def test_rmsnorm_forward_shape(self):
        norm = RMSNorm(dim=128)
        x = torch.randn(4, 10, 128) # Batch, Seq, Dim
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_rmsnorm_normalization_effect(self):
        dim = 32
        norm = RMSNorm(dim=dim, eps=1e-8) # Use smaller eps for more precise check

        # Test with default weight (ones)
        x = torch.randn(2, 5, dim) * 10 # Input with some variance
        out_default_weight = norm(x)

        # The RMS of the output (before learnable weight) should be close to 1
        # output = (x * rsqrt(mean(x^2) + eps)) * weight
        # So, (output / weight) should have RMS of approx 1
        # Since default weight is ones, normalized_part is effectively out_default_weight here.
        normalized_part_default_w = out_default_weight / norm.weight
        rms_of_normalized_part_default_w = torch.sqrt(normalized_part_default_w.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms_of_normalized_part_default_w, torch.ones_like(rms_of_normalized_part_default_w), atol=1e-5),
                        f"RMS of normalized output (default weight=1) not close to 1. Got: {rms_of_normalized_part_default_w}")

        # Test with a different learnable weight
        norm.weight.data.fill_(3.0) # Set weight to 3.0
        out_scaled_weight = norm(x)
        # (out_scaled_weight / new_weight) should still have RMS of approx 1
        normalized_part_scaled = out_scaled_weight / norm.weight
        rms_of_normalized_part_scaled = torch.sqrt(normalized_part_scaled.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms_of_normalized_part_scaled, torch.ones_like(rms_of_normalized_part_scaled), atol=1e-5),
                        f"RMS of normalized output (with weight=3) not close to 1. Got: {rms_of_normalized_part_scaled}")

        # Check that the output is scaled by the weight relative to the default weight output
        # out_scaled_weight should be approx 3 * (output if weight was 1)
        # We can use out_default_weight (where effective weight was 1) for comparison
        self.assertTrue(torch.allclose(out_scaled_weight, out_default_weight * 3.0, atol=1e-5),
                        "Output with weight=3 not correctly scaled relative to output with weight=1.")


    def test_rmsnorm_eps_effect(self):
        dim = 16
        # Using very small values for x to make eps effect more prominent
        x = torch.randn(1, 1, dim) * 1e-4

        norm_large_eps = RMSNorm(dim=dim, eps=1.0)
        out_large_eps = norm_large_eps(x)

        norm_small_eps = RMSNorm(dim=dim, eps=1e-12)
        out_small_eps = norm_small_eps(x)

        # If x is very small, x.pow(2).mean() can be smaller than eps.
        # For large_eps, denominator approx sqrt(eps). For small_eps, denominator approx sqrt(mean(x^2)).
        # Thus, outputs should differ significantly if mean(x^2) is not >> eps.
        self.assertFalse(torch.allclose(out_large_eps, out_small_eps, atol=1e-5),
                         "Output with large and small eps should differ for small inputs if eps has an effect.")

    def test_rmsnorm_backward_pass_and_grad(self):
        dim = 8
        norm = RMSNorm(dim=dim)
        x = torch.randn(2, 3, dim, requires_grad=True)

        self.assertTrue(norm.weight.requires_grad)

        out = norm(x)
        loss = out.mean() # Dummy loss
        loss.backward()

        self.assertIsNotNone(x.grad, "Input x should have gradients.")
        self.assertTrue(torch.is_tensor(x.grad)) # Ensure grad is a tensor
        self.assertNotEqual(x.grad.abs().sum().item(), 0, "Sum of absolute gradients for x should not be zero.")


        self.assertIsNotNone(norm.weight.grad, "RMSNorm learnable weight should have gradients.")
        self.assertTrue(torch.is_tensor(norm.weight.grad)) # Ensure grad is a tensor
        self.assertNotEqual(norm.weight.grad.abs().sum().item(), 0, "Sum of absolute gradients for weight should not be zero.")


if __name__ == '__main__':
    unittest.main()
