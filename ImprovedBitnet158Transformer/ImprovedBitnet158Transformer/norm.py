# Placeholder for Normalization modules
# This file will typically contain implementations like RMSNorm (Root Mean Square Normalization).

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Reference: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Learnable gain, initialized to 1

    def _norm(self, x):
        # Calculate Root Mean Square: sqrt(mean of squares)
        # Add eps for numerical stability before sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize the input
        normalized_x = self._norm(x)
        # Scale with learnable gain
        return self.weight * normalized_x

if __name__ == '__main__':
    # Example Usage (placeholder)
    # Parameters
    batch_size = 4
    seq_len = 10
    model_dim = 128

    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, model_dim)

    # Initialize RMSNorm layer
    rmsnorm_layer = RMSNorm(dim=model_dim)

    # Forward pass
    output = rmsnorm_layer(dummy_input)

    print("RMSNorm input shape:", dummy_input.shape)
    print("RMSNorm output shape:", output.shape)

    # Check if output statistics are as expected (mean close to 0, std close to gain after norm)
    # Note: RMSNorm doesn't center the data (no learnable bias like LayerNorm)
    # So, mean won't necessarily be 0.
    # The variance of the _norm(x) part should be close to 1.
    # The variance of the output will be close to self.weight.pow(2).mean() if weight is not scalar.
    # For a scalar weight (or all elements equal), var(output) ~ weight^2 * var(_norm(x))

    # Calculate variance of the normalized part (before scaling by weight)
    # To do this, we can call the internal _norm method.
    normalized_part = rmsnorm_layer._norm(dummy_input)
    print("Mean of normalized_part (should be somewhat close to 0):", normalized_part.mean().item())
    print("Std of normalized_part (should be close to 1):", normalized_part.std().item())

    # Check learnable parameter
    print("\nLearnable gain (self.weight) initial values (first 5):", rmsnorm_layer.weight.data[:5])
    # After training, these weights would be updated.

    # Example with a different gain initialization for testing
    custom_gain = torch.arange(1, model_dim + 1, dtype=torch.float32) / model_dim
    rmsnorm_layer_custom_gain = RMSNorm(dim=model_dim)
    rmsnorm_layer_custom_gain.weight.data = custom_gain.clone() # Assign new gain

    output_custom_gain = rmsnorm_layer_custom_gain(dummy_input)
    # The std of output_custom_gain should reflect the new gains.
    # This is harder to verify with a simple print, but shows how the gain works.
    print("\nOutput std with custom gain (example):", output_custom_gain.std().item())
    # The actual std will depend on the interaction of input data and the specific gain vector.
    # If gain was a scalar 'g', output std would be approx 'g' * input_normalized_std.
    # With a vector gain, it's more complex.

    print(f"\nRMSNorm with dim={model_dim} initialized.")
