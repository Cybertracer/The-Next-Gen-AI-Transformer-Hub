import torch

def absmean_quant(w: torch.Tensor,
                  group_size: int = 128,
                  ternary_method: str = 'round' # options: 'round', 'threshold'
                  ) -> torch.Tensor:
    """Group-wise or per-tensor AbsMean quantization for weights.
    Returns dequantized output to simulate quantization effect.
    Weights are quantized to {-1, 0, +1} relative to their scale.

    Args:
        w (torch.Tensor): Weight tensor.
        group_size (int, optional): Size of groups for group-wise quantization.
                                    If None or <=0, per-tensor quantization is used. Defaults to 128.
        ternary_method (str, optional): Method for ternary quantization.
                                        'round': Rounds to nearest {-1, 0, 1}.
                                        'threshold': Uses a 0.5 threshold on normalized magnitude.
                                        Defaults to 'round'.
    Returns:
        torch.Tensor: Dequantized weights after applying ternary quantization.
    """

    if w.numel() == 0:
        return w.clone() # Return empty tensor if input is empty

    # Per-tensor quantization path (also handles w.numel() <= group_size)
    if group_size is None or group_size <= 0 or w.numel() <= group_size:
        scale = w.abs().mean().clamp_min(1e-9) # clamp_min to avoid division by zero
        normalized_w = w / scale

        if ternary_method == 'round':
            q_w = normalized_w.round().clamp(-1, 1)
        elif ternary_method == 'threshold':
            # Values with |normalized_w| <= 0.5 become 0, others +/-1 based on sign.
            q_w = torch.sign(normalized_w) * (normalized_w.abs() > 0.5).float()
        else:
            raise ValueError(f"Unknown ternary_method: {ternary_method}. Choose 'round' or 'threshold'.")

        dequantized_w = q_w * scale
        return dequantized_w

    # Group-wise quantization path
    else:
        orig_shape = w.shape

        # Fallback for tensors not perfectly divisible by group_size (and larger than group_size)
        if w.numel() % group_size != 0:
            # print(f"Warning: tensor size {w.numel()} not perfectly divisible by group_size {group_size}. Falling back to per-tensor.")
            scale = w.abs().mean().clamp_min(1e-9)
            normalized_w = w / scale
            if ternary_method == 'round':
                q_w = normalized_w.round().clamp(-1, 1)
            elif ternary_method == 'threshold':
                q_w = torch.sign(normalized_w) * (normalized_w.abs() > 0.5).float()
            else:
                raise ValueError(f"Unknown ternary_method: {ternary_method}")
            return (q_w * scale) # Return directly, no reshape needed as it's effectively per-tensor

        w_reshaped = w.view(-1, group_size)
        scale_grouped = w_reshaped.abs().mean(dim=-1, keepdim=True).clamp_min(1e-9)

        normalized_w_grouped = w_reshaped / scale_grouped

        if ternary_method == 'round':
            q_w_grouped = normalized_w_grouped.round().clamp(-1, 1)
        elif ternary_method == 'threshold':
            q_w_grouped = torch.sign(normalized_w_grouped) * (normalized_w_grouped.abs() > 0.5).float()
        else:
            raise ValueError(f"Unknown ternary_method: {ternary_method}")

        dequantized_w_grouped = q_w_grouped * scale_grouped
        return dequantized_w_grouped.view(orig_shape)


if __name__ == '__main__':
    print("Running absmean_quant.py example with different ternary methods:")

    weights_example = torch.tensor([[-1.5, -0.8, -0.2, 0.2, 0.8, 1.5],
                                    [-0.6, -0.4, -0.1, 0.1, 0.4, 0.6]]) * 2.0
    print(f"Original weights:\n{weights_example}")

    # Per-tensor, 'round' method
    dequant_pt_round = absmean_quant(weights_example, group_size=None, ternary_method='round')
    print(f"\nDequantized (per-tensor, 'round'):\n{dequant_pt_round}")
    scale_pt_r = weights_example.abs().mean().clamp_min(1e-9)
    print(f"Effective ternary (per-tensor, 'round'):\n{(weights_example/scale_pt_r).round().clamp(-1,1)}")


    # Per-tensor, 'threshold' method
    dequant_pt_thresh = absmean_quant(weights_example, group_size=None, ternary_method='threshold')
    print(f"\nDequantized (per-tensor, 'threshold'):\n{dequant_pt_thresh}")
    scale_pt_t = weights_example.abs().mean().clamp_min(1e-9)
    norm_w_pt_t = weights_example/scale_pt_t
    print(f"Effective ternary (per-tensor, 'threshold'):\n{torch.sign(norm_w_pt_t) * (norm_w_pt_t.abs() > 0.5).float()}")

    # Group-wise example
    weights_gw = torch.randn(2, 12) * 3 # 2*12 = 24 elements
    group_size_gw = 6 # 24 / 6 = 4 groups

    print(f"\nOriginal weights (for group-wise, shape {weights_gw.shape}):\n{weights_gw}")

    # Group-wise, 'round'
    dequant_gw_round = absmean_quant(weights_gw, group_size=group_size_gw, ternary_method='round')
    print(f"Dequantized (group-wise, 'round', group_size={group_size_gw}):\n{dequant_gw_round}")

    # Group-wise, 'threshold'
    dequant_gw_thresh = absmean_quant(weights_gw, group_size=group_size_gw, ternary_method='threshold')
    print(f"Dequantized (group-wise, 'threshold', group_size={group_size_gw}):\n{dequant_gw_thresh}")

    # Test with all zeros
    all_zeros = torch.zeros(2,6)
    print(f"\nOriginal all zeros: {all_zeros}")
    print(f"Dequantized all zeros (per-tensor, round): {absmean_quant(all_zeros, group_size=None, ternary_method='round')}")
    print(f"Dequantized all zeros (per-tensor, threshold): {absmean_quant(all_zeros, group_size=None, ternary_method='threshold')}")
    print(f"Dequantized all zeros (group-wise, round): {absmean_quant(all_zeros, group_size=3, ternary_method='round')}")
    print(f"Dequantized all zeros (group-wise, threshold): {absmean_quant(all_zeros, group_size=3, ternary_method='threshold')}")

    # Test fallback for non-divisible group size
    weights_nd = torch.randn(1, 7) * 2 # 7 elements, group size 3
    print(f"\nOriginal non-divisible weights (shape {weights_nd.shape}):\n{weights_nd}")
    dequant_nd_round = absmean_quant(weights_nd, group_size=3, ternary_method='round')
    print(f"Dequantized non-divisible (group_size=3, 'round', should fallback to per-tensor):\n{dequant_nd_round}")
    # Verify it's same as per-tensor
    dequant_nd_pt_round = absmean_quant(weights_nd, group_size=None, ternary_method='round')
    print(f"Dequantized non-divisible (per-tensor for comparison):\n{dequant_nd_pt_round}")
    assert torch.allclose(dequant_nd_round, dequant_nd_pt_round), "Fallback for non-divisible group size did not match per-tensor."
    print("Fallback for non-divisible group size verified.")
