import torch

def absmean_quant(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Group-wise or per-tensor AbsMean quantization for weights.
    Returns dequantized output to simulate quantization effect.
    Weights are quantized to {-1, 0, +1} relative to their scale.
    """
    if group_size is None or group_size <= 0:  # Per-tensor quantization
        # Clamp scale to avoid issues with all-zero tensors.
        # If w.abs().mean() is 0, scale becomes 1e-6.
        # normalized_w will be 0. q_w will be 0. Result will be 0. Correct.
        scale = w.abs().mean().clamp_min(1e-6)
        normalized_w = w / scale
        q_w = normalized_w.round().clamp(-1, 1) # Ternary values based on rounding
    else:  # Group-wise quantization
        orig_shape = w.shape
        if w.numel() == 0:
            return w.clone() # Return empty tensor if input is empty
        if w.numel() % group_size != 0 and w.numel() > group_size :
            # This basic version requires group_size to perfectly divide the tensor numel
            # or for the tensor to be smaller than group_size (becomes per-tensor like)
            # For simplicity, we'll fall back to per-tensor if not perfectly divisible and large.
            # A more robust implementation might pad or handle remainders.
            # print(f"Warning: tensor size {w.numel()} not perfectly divisible by group_size {group_size}. Falling back to per-tensor.")
            scale = w.abs().mean().clamp_min(1e-6)
            normalized_w = w / scale
            q_w = normalized_w.round().clamp(-1, 1) # Ternary values
            return q_w * scale


        # If tensor numel is less than group_size, treat as a single group (per-tensor)
        if w.numel() <= group_size:
             scale = w.abs().mean().clamp_min(1e-6)
             normalized_w = w / scale
             q_w = normalized_w.round().clamp(-1, 1)
             return q_w * scale


        w_reshaped = w.view(-1, group_size)
        scale_grouped = w_reshaped.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)

        normalized_w_grouped = w_reshaped / scale_grouped
        q_w_grouped = normalized_w_grouped.round().clamp(-1, 1) # Ternary values

        dequantized_w_grouped = q_w_grouped * scale_grouped
        return dequantized_w_grouped.view(orig_shape)

    # For per-tensor, dequantize here
    dequantized_w = q_w * scale
    return dequantized_w

if __name__ == '__main__':
    print("Running absmean_quant.py example:")

    # Per-tensor example
    weights_pt = torch.tensor([[-1.5, -0.8, -0.2],
                               [ 0.0,  0.1,  0.6],
                               [ 1.2,  1.7,  0.3]]) * 2.0
    print(f"\nOriginal weights (for per-tensor):\n{weights_pt}")
    dequant_weights_pt = absmean_quant(weights_pt, group_size=None)
    print(f"Dequantized weights after per-tensor absmean_quant:\n{dequant_weights_pt}")
    # Show the effective ternary weights (not directly returned but useful to see)
    scale_pt_eff = weights_pt.abs().mean().clamp_min(1e-6)
    ternary_pt_eff = (weights_pt / scale_pt_eff).round().clamp(-1,1)
    print(f"Effective ternary weights (for per-tensor):\n{ternary_pt_eff}")


    # Group-wise example
    # Make sure numel is divisible by group_size for this simple example version
    weights_gw = torch.randn(4, 16) * 5 # 4*16 = 64 elements
    group_size_gw = 16 # 64 / 16 = 4 groups

    print(f"\nOriginal weights (for group-wise, shape {weights_gw.shape}):\n{weights_gw}")
    dequant_weights_gw = absmean_quant(weights_gw, group_size=group_size_gw)
    print(f"Dequantized weights after group-wise absmean_quant (group_size={group_size_gw}):\n{dequant_weights_gw}")

    # Example of how effective ternary weights would look for group-wise (for one group)
    first_group = weights_gw.view(-1, group_size_gw)[0]
    scale_gw_eff_group0 = first_group.abs().mean().clamp_min(1e-6)
    ternary_gw_eff_group0 = (first_group / scale_gw_eff_group0).round().clamp(-1,1)
    print(f"Effective ternary weights for first group (group_size={group_size_gw}):\n{ternary_gw_eff_group0}")
    print(f"Original first group:\n{first_group}")
    print(f"Dequantized first group (from output):\n{dequant_weights_gw.view(-1,group_size_gw)[0]}")

    # Test with all zeros
    all_zeros = torch.zeros(2,32)
    print(f"\nOriginal all zeros: {all_zeros}")
    dequant_zeros_pt = absmean_quant(all_zeros, group_size=None)
    dequant_zeros_gw = absmean_quant(all_zeros, group_size=16)
    print(f"Dequantized all zeros (per-tensor): {dequant_zeros_pt}")
    print(f"Dequantized all zeros (group-wise): {dequant_zeros_gw}")
