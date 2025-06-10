import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # The gamma parameter (scale) is learnable
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization formula."""
        # x.pow(2).mean(-1, keepdim=True) computes E[x^2] along the last dimension
        # torch.rsqrt is 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        Input tensor is expected to have shape (batch_size, seq_len, dim)
        or any shape where the last dimension is 'dim'.
        """
        output = self._norm(x.float()).type_as(x) # Convert to float for norm, then back to original type
        return output * self.weight

if __name__ == '__main__':
    print("Running norm.py (RMSNorm) example:")

    # Parameters
    batch_size = 4
    seq_len = 10
    feature_dim = 64

    # Create RMSNorm layer
    rms_norm_layer = RMSNorm(dim=feature_dim)
    print(f"RMSNorm layer: {rms_norm_layer}")

    # Create a sample input tensor
    input_tensor = torch.randn(batch_size, seq_len, feature_dim) * 5 # Add some scale
    print(f"Input tensor shape: {input_tensor.shape}")

    # Forward pass
    output_tensor = rms_norm_layer(input_tensor)
    print(f"Output tensor shape: {output_tensor.shape}")

    # Check statistics of the output (mean should be close to 0, std close to 1 before scaling by weight)
    # For RMSNorm, the mean of (output / weight) is not necessarily 0, but its RMS is 1.
    normalized_output_rms = torch.sqrt((output_tensor / rms_norm_layer.weight).pow(2).mean(-1))
    print(f"RMS of (output / weight) along last dim (should be close to 1.0):\n{normalized_output_rms}")

    # Check that weights are being applied
    rms_norm_layer.weight.data.fill_(2.0) # Set all weights to 2.0
    output_tensor_scaled = rms_norm_layer(input_tensor)
    # RMS of (output_tensor_scaled / new_weight) should still be ~1.0
    # and output_tensor_scaled should be approx 2 * (output_tensor when weight was 1)
    normalized_output_scaled_rms = torch.sqrt((output_tensor_scaled / rms_norm_layer.weight).pow(2).mean(-1))
    print(f"RMS of (output_scaled / new_weight=2.0) along last dim (should be close to 1.0):\n{normalized_output_scaled_rms}")

    print(f"Output mean (sample): {output_tensor_scaled.mean().item()}") # Mean can be non-zero
    print(f"Output std (sample): {output_tensor_scaled.std().item()}")   # Std reflects the learned weight
