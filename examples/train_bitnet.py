# examples/train_bitnet.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time

# Adjust path to import from the parent directory's module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from ImprovedBitNet1p58Transformer.ImprovedBitNet1p58Transformer import (
        BitTransformer,
        QUANT_CONFIG,
        init_bitlinear # For applying custom init if desired
    )
except ImportError as e:
    print(f"Error importing from ImprovedBitNet1p58Transformer: {e}")
    print("Make sure you are in the 'examples/' directory or the repository root,")
    print("or that the ImprovedBitNet1p58Transformer package is installed correctly.")
    sys.exit(1)

def generate_dummy_data(num_samples, max_seq_len, vocab_size, target_seq_len):
    # Ensure target_seq_len for data matches model's max_seq_len or is appropriate
    # For LM, input and target might be shifted versions of the same sequence.
    # For classification, target might be a single class index.
    # This example is for next-token prediction style LM.

    # Input sequences: random integers from 0 to vocab_size-1
    # Shape: (num_samples, target_seq_len)
    input_data = torch.randint(0, vocab_size, (num_samples, target_seq_len))

    # Target sequences: shifted version of input_data for next-token prediction
    # Shape: (num_samples, target_seq_len)
    # For CrossEntropyLoss, targets should be class indices (long), not one-hot.
    # The loss will handle ignore_index for padding if needed.
    target_data = torch.roll(input_data, shifts=-1, dims=1)
    # Set the last token's target to something valid, e.g., a pad token or a specific token.
    # Or, ensure loss function ignores this last prediction if seq_len is fixed.
    # For simplicity, let's just use the rolled version. Loss on last token might be meaningless
    # if we don't have a specific target for it (e.g. an EOS token prediction).
    # A common practice is to make the target for the last input token a padding token,
    # and set ignore_index in the loss function.
    # Or, only predict up to seq_len-1 by adjusting logits and targets before loss calculation.

    return TensorDataset(input_data, target_data)

def main():
    parser = argparse.ArgumentParser(description="Example Training Script for BitTransformer")

    # Model Parameters
    parser.add_argument('--num_tokens', type=int, default=1000, help='Vocabulary size.')
    parser.add_argument('--dim', type=int, default=64, help='Model dimension.')
    parser.add_argument('--depth', type=int, default=2, help='Number of transformer blocks.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--mlp_dim_multiplier', type=int, default=4, help='Multiplier for MLP hidden dim.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes (vocab size for LM).')
    parser.add_argument('--max_seq_len', type=int, default=32, help='Maximum sequence length for model.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # Quantization Parameters
    q_conf = QUANT_CONFIG
    parser.add_argument('--activation_bits', type=int, default=q_conf.get('activation_bits', 8), help='Bits for activation quantization.')
    parser.add_argument('--weight_group_size', type=int, default=q_conf.get('weight_group_size', 128), help='Group size for weight quantization.')
    parser.add_argument('--quantize_embedding', type=bool, default=q_conf.get('quantize_embeddings', False), help='Whether to use BitEmbedding.')
    parser.add_argument('--embedding_activation_bits', type=int, default=q_conf.get('activation_bits', 8))
    parser.add_argument('--embedding_stochastic_rounding', type=bool, default=False)

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.') # Adjusted for small model/data
    parser.add_argument('--dummy_data_size', type=int, default=64, help='Number of samples in dummy dataset.')
    parser.add_argument('--seq_len_data', type=int, default=32, help='Sequence length for dummy data.')
    parser.add_argument('--save_checkpoint_path', type=str, default='./bittransformer_checkpoint.pth', help='Path to save model checkpoint.')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to load model checkpoint from.')
    parser.add_argument('--print_interval', type=int, default=5, help='Interval for printing training loss.')

    args = parser.parse_args()

    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 30)

    # 1. Setup device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create dummy dataset and dataloader
    if args.seq_len_data != args.max_seq_len:
        print(f"Warning: seq_len_data ({args.seq_len_data}) for dummy data generation does not match model's max_seq_len ({args.max_seq_len}). "
              f"Adjusting seq_len_data to {args.max_seq_len}.")
        args.seq_len_data = args.max_seq_len

    dummy_dataset = generate_dummy_data(args.dummy_data_size, args.max_seq_len, args.num_tokens, args.seq_len_data)
    train_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Instantiate model
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

    # model.apply(init_bitlinear) # Optional: Apply custom initialization

    # 4. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.load_checkpoint_path and os.path.exists(args.load_checkpoint_path):
        print(f"Loading checkpoint from {args.load_checkpoint_path}")
        checkpoint = torch.load(args.load_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1 # epoch in checkpoint is last completed epoch
        print(f"Resuming training from epoch {start_epoch}")

    # 5. Training loop
    print("\nStarting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        start_time = time.time()
        for i, (batch_inputs, batch_targets) in enumerate(train_loader):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()

            attn_mask = None # Assuming no padding in dummy data for simplicity

            logits = model(batch_inputs, mask=attn_mask)

            loss = criterion(logits.view(-1, args.num_classes), batch_targets.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % args.print_interval == 0 or (i + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # 6. Checkpoint saving
        if args.save_checkpoint_path:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'args': args
            }
            # Save epoch-specific checkpoint
            epoch_save_path = f"{os.path.splitext(args.save_checkpoint_path)[0]}_epoch{epoch+1}.pth"
            torch.save(checkpoint_data, epoch_save_path)
            print(f"Checkpoint saved to {epoch_save_path}")

            # Save latest checkpoint (overwrites previous latest)
            latest_save_path = args.save_checkpoint_path + ".latest"
            torch.save(checkpoint_data, latest_save_path)
            print(f"Latest checkpoint saved to {latest_save_path}")

    print("\nTraining complete.")
    if args.save_checkpoint_path:
        print(f"Final model checkpoint (latest) available at {args.save_checkpoint_path}.latest")

if __name__ == '__main__':
    main()
