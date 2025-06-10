# TestImprovedBitNet1p58Transformer/TestImprovedBitNet1p58Transformer/test_bit_transformer.py
import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import (
        BitTransformer,
        BitTransformerBlock
    )
    # Import supporting classes to check types or pass as args
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bitnet_quant import BitEmbedding, QUANT_CONFIG, BitLinear
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bit_attention import BitAttention
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.bit_feedforward import BitFeedForward
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer.norm import RMSNorm


except ImportError as e:
    print(f"Error importing modules for testing in test_bit_transformer.py: {e}")
    raise

class TestBitTransformer(unittest.TestCase):

    def setUp(self):
        # Common parameters for tests
        self.test_num_tokens = 100
        self.test_dim = 32 # Must be divisible by heads
        self.test_depth = 2
        self.test_heads = 4
        self.test_dim_head = self.test_dim // self.test_heads
        self.test_mlp_dim = self.test_dim * 2
        self.test_num_classes = self.test_num_tokens
        self.test_max_seq_len = 20
        self.test_batch_size = 2
        self.test_seq_len = 10

        self.default_model_args = {
            'num_tokens': self.test_num_tokens,
            'dim': self.test_dim,
            'depth': self.test_depth,
            'heads': self.test_heads,
            'dim_head': self.test_dim_head,
            'mlp_dim': self.test_mlp_dim,
            'num_classes': self.test_num_classes,
            'max_seq_len': self.test_max_seq_len
        }
        self.sample_input_ids = torch.randint(0, self.test_num_tokens,
                                              (self.test_batch_size, self.test_seq_len))

    def test_bittransformerblock_instantiation(self):
        block = BitTransformerBlock(
            dim=self.test_dim,
            heads=self.test_heads,
            dim_head=self.test_dim_head,
            mlp_dim=self.test_mlp_dim
        )
        self.assertIsInstance(block, nn.Module)
        self.assertIsInstance(block.attention, BitAttention)
        self.assertIsInstance(block.feed_forward, BitFeedForward)
        self.assertIsInstance(block.norm1, RMSNorm)
        self.assertIsInstance(block.norm2, RMSNorm)


    def test_bittransformerblock_forward_shape(self):
        block = BitTransformerBlock(
            dim=self.test_dim, heads=self.test_heads,
            dim_head=self.test_dim_head, mlp_dim=self.test_mlp_dim
        )
        x = torch.randn(self.test_batch_size, self.test_seq_len, self.test_dim)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_bittransformerblock_forward_with_mask(self):
        block = BitTransformerBlock(
            dim=self.test_dim, heads=self.test_heads,
            dim_head=self.test_dim_head, mlp_dim=self.test_mlp_dim
        )
        block.eval()
        x = torch.randn(self.test_batch_size, self.test_seq_len, self.test_dim)
        # Mask where 1 means keep, 0 means mask out.
        # BitAttention's masked_fill expects mask == 0 for positions to fill.
        mask = torch.ones(self.test_batch_size, self.test_seq_len).bool()
        if self.test_seq_len > 1: mask[0, -1] = False

        out = block(x, mask=mask)
        self.assertEqual(out.shape, x.shape)

    def test_bittransformer_instantiation(self):
        model = BitTransformer(**self.default_model_args)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(len(model.layers), self.test_depth)
        self.assertIsInstance(model.embedding, nn.Embedding) # Default

    def test_bittransformer_quantize_embedding(self):
        args_quant_emb = self.default_model_args.copy()
        args_quant_emb['quantize_embedding'] = True
        model = BitTransformer(**args_quant_emb)
        self.assertIsInstance(model.embedding, BitEmbedding)


    def test_bittransformer_forward_shape(self):
        model = BitTransformer(**self.default_model_args)
        out = model(self.sample_input_ids)
        expected_shape = (self.test_batch_size, self.test_seq_len, self.test_num_classes)
        self.assertEqual(out.shape, expected_shape)

    def test_bittransformer_forward_with_mask(self):
        model = BitTransformer(**self.default_model_args)
        model.eval()
        mask = torch.ones_like(self.sample_input_ids).bool()
        if self.sample_input_ids.shape[1] > 1: mask[:, -1] = False

        out = model(self.sample_input_ids, mask=mask)
        expected_shape = (self.test_batch_size, self.test_seq_len, self.test_num_classes)
        self.assertEqual(out.shape, expected_shape)

    def test_bittransformer_max_seq_len_handling(self):
        model = BitTransformer(**self.default_model_args)
        long_input_ids = torch.randint(0, self.test_num_tokens,
                                       (self.test_batch_size, self.test_max_seq_len + 5))
        with self.assertRaisesRegex(ValueError, "Input sequence length .* exceeds maximum positional embedding length"):
            model(long_input_ids)

        exact_len_input_ids = torch.randint(0, self.test_num_tokens,
                                            (self.test_batch_size, self.test_max_seq_len))
        try:
            out = model(exact_len_input_ids)
            self.assertEqual(out.shape, (self.test_batch_size, self.test_max_seq_len, self.test_num_classes))
        except Exception as e:
            self.fail(f"Forward pass with exact max_seq_len failed: {e}")

    def test_bittransformer_different_configs(self):
        args_deep = self.default_model_args.copy()
        args_deep['depth'] = 4
        model_deep = BitTransformer(**args_deep)
        self.assertEqual(len(model_deep.layers), 4)
        out_deep = model_deep(self.sample_input_ids)
        self.assertEqual(out_deep.shape, (self.test_batch_size, self.test_seq_len, self.test_num_classes))

        args_quant_params = self.default_model_args.copy()
        custom_act_bits = 4
        custom_weight_group_size = 16
        args_quant_params['activation_bits'] = custom_act_bits
        args_quant_params['weight_group_size'] = custom_weight_group_size

        model_q_params = BitTransformer(**args_quant_params)

        first_block = model_q_params.layers[0]
        self.assertIsInstance(first_block, BitTransformerBlock)
        first_attention_layer = first_block.attention
        self.assertIsInstance(first_attention_layer, BitAttention)
        q_proj_layer = first_attention_layer.to_q
        self.assertIsInstance(q_proj_layer, BitLinear)

        self.assertEqual(q_proj_layer.activation_bits, custom_act_bits)
        self.assertEqual(q_proj_layer.group_size, custom_weight_group_size)

        ff_layer_in_block = first_block.feed_forward
        self.assertIsInstance(ff_layer_in_block, BitFeedForward)
        # Assuming BitFeedForward.net[0] is the first BitLinear layer
        first_linear_in_ff = ff_layer_in_block.net[0]
        self.assertIsInstance(first_linear_in_ff, BitLinear)
        self.assertEqual(first_linear_in_ff.activation_bits, custom_act_bits)
        self.assertEqual(first_linear_in_ff.group_size, custom_weight_group_size)


        out_q_params = model_q_params(self.sample_input_ids)
        self.assertEqual(out_q_params.shape, (self.test_batch_size, self.test_seq_len, self.test_num_classes))


if __name__ == '__main__':
    unittest.main()
