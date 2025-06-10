# ImprovedBitNet1p58Transformer/ImprovedBitNet1p58Transformer/__init__.py

"""
Main package for the ImprovedBitNet1p58Transformer.

This package provides the core components for building and using
the BitNet 1.58-bit Transformer model, including quantization utilities,
quantized layers, and the main model architecture.
"""

# Import from the bitnet_quant submodule to make them available at this level
from .bitnet_quant import (
    absmax_quant,
    quantize_activations,
    absmean_quant,
    BitLinear,
    QUANT_CONFIG,
    init_bitlinear
)

# Placeholder imports for other potential top-level components
# (uncomment and adjust as these modules are developed)
# from .bit_attention import BitAttention
# from .bit_feedforward import BitFeedForward
# from .norm import RMSNorm # Or other normalization layers
# from .bit_transformer import BitTransformer, BitTransformerBlock

# Define what gets imported with "from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import *"
# This should be curated to expose the main public API.
__all__ = [
    # From bitnet_quant
    'absmax_quant',
    'quantize_activations',
    'absmean_quant',
    'BitLinear',
    'QUANT_CONFIG',
    'init_bitlinear',

    # Placeholders for other components when they are ready
    # 'BitAttention',
    # 'BitFeedForward',
    # 'RMSNorm',
    # 'BitTransformer',
    # 'BitTransformerBlock',
]

print("ImprovedBitNet1p58Transformer package loaded.") # Optional: for debugging imports
