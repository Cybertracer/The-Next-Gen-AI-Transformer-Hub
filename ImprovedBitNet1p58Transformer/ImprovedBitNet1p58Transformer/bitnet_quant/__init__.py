# ImprovedBitNet1p58Transformer/ImprovedBitNet1p58Transformer/bitnet_quant/__init__.py

"""
This package provides modules for quantization, including:
- Activation quantization (AbsMax)
- Weight quantization (AbsMean for ternary weights)
- Quantized layer implementations (e.g., BitLinear, BitEmbedding)
- Utility functions and configurations for quantization.
"""

from .absmax_quant import absmax_quant, quantize_activations
from .absmean_quant import absmean_quant
from .quant_layers import BitLinear, BitEmbedding # Added BitEmbedding
from .quant_utils import QUANT_CONFIG, init_bitlinear

# Define what gets imported with "from .bitnet_quant import *"
__all__ = [
    'absmax_quant',
    'quantize_activations',
    'absmean_quant',
    'BitLinear',
    'BitEmbedding', # Added BitEmbedding
    'QUANT_CONFIG',
    'init_bitlinear'
]

# print("bitnet_quant package loaded.") # Optional
