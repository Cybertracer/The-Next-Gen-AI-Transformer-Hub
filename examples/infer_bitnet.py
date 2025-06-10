# examples/infer_bitnet.py
import argparse
import torch
import torch.nn.functional as F # For softmax if sampling
import sys
import os

# Adjust path to import from the parent directory's module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import (
        BitTransformer,
        QUANT_CONFIG
        # init_bitlinear is not typically needed for inference if model is saved
    )
except ImportError as e:
    print(f"Error importing from ImprovedBitNet1p58Transformer: {e}")
    print("Make sure you are in the 'examples/' directory or the repository root,")
    print("or that the ImprovedBitNet1p58Transformer package is installed correctly.")
    sys.exit(1)

def generate_sequence(model, input_ids, max_gen_len, device, top_k=None):
    """
    Autoregressively generate a sequence of token IDs.
    Args:
        model: The BitTransformer model.
        input_ids: Tensor of shape (1, current_seq_len) with starting token IDs.
        max_gen_len: Maximum number of new tokens to generate.
        device: The device to run generation on.
        top_k: If not None, use top-k sampling. Otherwise, greedy.
    Returns:
        Tensor of shape (1, original_seq_len + generated_len)
    """
    model.eval() # Ensure model is in evaluation mode
    generated_ids = input_ids.clone().to(device) # (1, current_seq_len)

    with torch.no_grad():
        for _ in range(max_gen_len):
            # Get logits for the last token in the current sequence
            # Input to model should be (batch_size, seq_len)
            # Logits will be (batch_size, seq_len, num_classes)
            current_seq_for_model = generated_ids

            # Attention mask: None for now, assuming no padding in generation after prompt.
            # If prompt itself could have padding, a mask would be needed for the prompt part.
            attn_mask = None

            logits = model(current_seq_for_model, mask=attn_mask) # (1, current_seq_len, num_classes)

            # Get logits for the very last token position
            next_token_logits = logits[:, -1, :] # (1, num_classes)

            if top_k is not None and top_k > 0:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                probabilities = F.softmax(top_k_logits, dim=-1) # (1, top_k)
                next_token_idx_in_top_k = torch.multinomial(probabilities, num_samples=1) # (1, 1)
                next_token_id = top_k_indices.gather(-1, next_token_idx_in_top_k) # (1,1)
            else:
                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # (1,1)

            # Append the predicted token ID to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Optional: Add EOS token handling here if you have one defined
            # if next_token_id.item() == EOS_TOKEN_ID:
            #     break

    return generated_ids


def main():
    parser = argparse.ArgumentParser(description="Example Inference Script for BitTransformer")

    # Model Checkpoint
    parser.add_argument('--load_checkpoint_path', type=str, required=True, help='Path to load model checkpoint from.')

    # Model Parameters (should match the loaded checkpoint's model args)
    # It's often better to save these with the checkpoint and load them,
    # but for this example, we require them as CLI args.
    # Defaults are set to match the training script's reduced defaults for dummy testing.
    parser.add_argument('--num_tokens', type=int, default=1000, help='Vocabulary size.')
    parser.add_argument('--dim', type=int, default=64, help='Model dimension.')
    parser.add_argument('--depth', type=int, default=2, help='Number of transformer blocks.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--mlp_dim_multiplier', type=int, default=4, help='Multiplier for MLP hidden dim.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes (vocab size for LM).')
    parser.add_argument('--max_seq_len', type=int, default=32, help='Maximum sequence length for model.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (for model instantiation, not active in eval).')

    # Quantization Parameters
    q_conf = QUANT_CONFIG
    parser.add_argument('--activation_bits', type=int, default=q_conf.get('activation_bits', 8))
    parser.add_argument('--weight_group_size', type=int, default=q_conf.get('weight_group_size', 128))
    parser.add_argument('--quantize_embedding', type=bool, default=q_conf.get('quantize_embeddings', False))
    parser.add_argument('--embedding_activation_bits', type=int, default=q_conf.get('activation_bits', 8))
    parser.add_argument('--embedding_stochastic_rounding', type=bool, default=False)


    # Inference Parameters
    parser.add_argument('--input_ids', type=str, default="10,20,30", help='Comma-separated string of input token IDs (e.g., "10,20,30").')
    parser.add_argument('--max_gen_len', type=int, default=20, help='Maximum number of new tokens to generate.')
    parser.add_argument('--top_k', type=int, default=None, help='If set, use top-k sampling with k=value. Default is greedy.')


    args = parser.parse_args()

    print("Inference Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model
    dim_head = args.dim // args.heads
    if args.dim % args.heads != 0:
        raise ValueError("Model dimension 'dim' must be divisible by 'heads'.")
    mlp_dim = args.dim * args.mlp_dim_multiplier

    model = BitTransformer(
        num_tokens=args.num_tokens,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        activation_bits=args.activation_bits,
        weight_group_size=args.weight_group_size,
        quantize_embedding=args.quantize_embedding,
        embedding_activation_bits=args.embedding_activation_bits,
        embedding_stochastic_rounding=args.embedding_stochastic_rounding
    ).to(device)

    # Load checkpoint
    if not os.path.exists(args.load_checkpoint_path):
        print(f"Error: Checkpoint path '{args.load_checkpoint_path}' does not exist.")
        sys.exit(1)

    print(f"Loading checkpoint from {args.load_checkpoint_path}")
    checkpoint = torch.load(args.load_checkpoint_path, map_location=device)

    # TODO: Ideally, load model architecture args from checkpoint['args']
    # to prevent mismatch. For now, assume CLI args are correct for the checkpoint.
    # if 'args' in checkpoint:
    #     print("Loading model architecture from checkpoint args.")
    #     # Re-init model with checkpoint_args if they differ significantly
    #     # This requires careful handling if CLI args should override some checkpoint args.

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state_dict loaded successfully.")

    model.eval()

    # Prepare input sequence
    try:
        input_token_ids_list = [int(token_id_str.strip()) for token_id_str in args.input_ids.split(',')]
        if not input_token_ids_list:
            raise ValueError("Input IDs cannot be empty.")
        input_ids_tensor = torch.tensor([input_token_ids_list], dtype=torch.long).to(device)
    except ValueError as e:
        print(f"Error parsing input_ids: {e}. Please provide a comma-separated list of integers.")
        sys.exit(1)

    if input_ids_tensor.shape[1] >= args.max_seq_len:
        print(f"Warning: Length of input_ids ({input_ids_tensor.shape[1]}) is >= max_seq_len ({args.max_seq_len}). "
              "Model might not generate new tokens if prompt is too long for positional embeddings.")
        # Consider truncating or erroring based on desired behavior.
        # For now, let it proceed.

    print(f"\nInput token IDs: {input_ids_tensor.tolist()}")
    print(f"Generating up to {args.max_gen_len} new tokens...")

    # Generate sequence
    generated_output_ids = generate_sequence(model, input_ids_tensor, args.max_gen_len, device, top_k=args.top_k)

    print(f"\nGenerated token IDs (input + generated):")
    print(generated_output_ids.tolist())

    num_input_tokens = input_ids_tensor.shape[1]
    newly_generated_ids = generated_output_ids[:, num_input_tokens:]
    print(f"\nNewly generated token IDs:")
    print(newly_generated_ids.tolist())


if __name__ == '__main__':
    main()
