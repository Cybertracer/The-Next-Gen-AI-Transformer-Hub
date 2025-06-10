import torch
import torch.nn as nn

# It's good practice to import specific classes if they are used for type checking here,
# or ensure this file is imported only after quant_layers is available.
# For init_bitlinear, we need BitLinear.
from .quant_layers import BitLinear

# Configuration dictionary for quantization parameters
# This can be expanded and used throughout the model implementation.
QUANT_CONFIG = {
    'activation_bits': 8,       # Default bits for activation quantization
    'weight_bits_target': 1.58, # Informational: target for weight representation (e.g., ternary)
                                # Actual weight quantization to {-1,0,1} is handled by absmean_quant
    'weight_group_size': 128,   # Default group size for weight quantization
    'quantize_embeddings': True # Example flag for controlling embedding quantization
    # Add other configurations as needed, e.g., specific quantization for different layer types
}

def init_bitlinear(module: nn.Module) -> None:
    """
    Custom weight initialization for BitLinear layers.
    Uses Kaiming Normal for weights and zeros for biases if they exist.

    Args:
        module: The module to initialize. If it's a BitLinear layer, its weights/bias are initialized.
    """
    if isinstance(module, BitLinear):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Example of how this might be used (will not run directly without a model structure)
if __name__ == '__main__':
    print("Running quant_utils.py example:")

    print("QUANT_CONFIG:", QUANT_CONFIG)

    # Create a dummy BitLinear layer for the initialization example
    # This requires BitLinear to be defined and importable, which it is from .quant_layers
    print("\nInitializing a sample BitLinear layer:")
    sample_bitlinear_layer = BitLinear(in_features=64, out_features=128)
    print("Weight before init (sample):\n", sample_bitlinear_layer.weight.data[0, :5])
    if sample_bitlinear_layer.bias is not None:
        print("Bias before init (sample):\n", sample_bitlinear_layer.bias.data[:5])

    init_bitlinear(sample_bitlinear_layer)

    print("\nWeight after init (Kaiming Normal, sample):\n", sample_bitlinear_layer.weight.data[0, :5])
    if sample_bitlinear_layer.bias is not None:
        print("Bias after init (Zeros, sample):\n", sample_bitlinear_layer.bias.data[:5])

    # Example of applying to a whole model (dummy model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = BitLinear(10, 20)
            self.linear2 = nn.Linear(20, 5) # A standard linear layer
            self.custom_bitlinear = BitLinear(5,2)

        def init_weights(self):
            self.apply(init_bitlinear) # .apply traverses all submodules

    print("\nInitializing a dummy model containing BitLinear layers:")
    model = DummyModel()
    # Print weights of one BitLinear before custom init
    print("Model's custom_bitlinear weight before init (sample):\n", model.custom_bitlinear.weight.data[0, :5])
    model.init_weights()
    print("Model's custom_bitlinear weight after init (sample):\n", model.custom_bitlinear.weight.data[0, :5])
    # Standard nn.Linear would not be affected by init_bitlinear
    print("Model's standard nn.Linear weight (unaffected by init_bitlinear, default init):\n", model.linear2.weight.data[0, :5])
