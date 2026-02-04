"""
Training Pipeline for LLM From Scratch

This script trains the GPT model on text data. It handles:
- Loading and preparing training data
- Setting up the model, optimizer, and learning rate schedule
- The training loop with loss tracking
- Validation evaluation
- Saving model checkpoints

Key Concepts:
- Optimizer (AdamW): Updates model weights to minimize loss
- Learning Rate: Controls how big each weight update step is
- Epoch: One complete pass through all training data
- Batch: A subset of data processed together
- Gradient: Direction to adjust weights (computed via backpropagation)
- Backpropagation: Algorithm that computes gradients through the network

PyTorch Concepts Introduced:
- torch.optim.AdamW: Optimizer with weight decay regularization
- loss.backward(): Compute gradients via backpropagation
- optimizer.step(): Update weights using gradients
- optimizer.zero_grad(): Clear gradients before next batch
- model.train() / model.eval(): Toggle training/evaluation mode
- torch.save / torch.load: Save and load model checkpoints
"""

import os
import time
import json
import urllib.request

import torch
from tqdm import tqdm

from .tokenizer import Tokenizer
from .gpt_model import GPTConfig, GPTModel
from .dataset import create_dataloaders


def get_device() -> torch.device:
    """
    Select the best available device for training.

    Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_sample_data(data_dir: str) -> str:
    """
    Download a sample text file for training.

    Uses "The Verdict" by Edith Wharton from Project Gutenberg -
    a short story that's small enough for quick training demos.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "the_verdict.txt")

    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        print(f"Downloading sample training data...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except urllib.error.URLError:
            # Fallback for macOS SSL certificate issues
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"Using existing data: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return text


@torch.no_grad()
def evaluate(
    model: GPTModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate model on validation data.

    @torch.no_grad() disables gradient computation, saving memory
    and speeding up evaluation (we don't need gradients for inference).

    Args:
        model: The GPT model
        val_loader: Validation DataLoader
        device: Device to run on

    Returns:
        Average validation loss
    """
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0.0
    num_batches = 0

    for input_ids, target_ids in val_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        _, loss = model(input_ids, targets=target_ids)
        total_loss += loss.item()
        num_batches += 1

    model.train()  # Set back to training mode
    return total_loss / max(num_batches, 1)


def train(
    text: str | None = None,
    config: GPTConfig | None = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    seq_len: int = 256,
    save_dir: str = "models",
    eval_interval: int = 1,
):
    """
    Train the GPT model.

    Args:
        text: Training text. If None, downloads sample data.
        config: Model configuration. If None, uses a small trainable config.
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seq_len: Sequence length for training samples
        save_dir: Directory to save model checkpoints
        eval_interval: Evaluate every N epochs
    """
    print("=" * 60)
    print("GPT TRAINING")
    print("=" * 60)

    # ---- Setup ----
    device = get_device()
    print(f"\nDevice: {device}")

    # Load data
    if text is None:
        text = download_sample_data("data")

    print(f"Text length: {len(text):,} characters")

    # Initialize tokenizer
    tokenizer = Tokenizer()
    total_tokens = len(tokenizer.encode(text))
    print(f"Total tokens: {total_tokens:,}")

    # Model configuration
    # Using a smaller config that trains well on a laptop
    if config is None:
        config = GPTConfig(
            vocab_size=50257,
            max_seq_len=seq_len,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
        )

    print(f"\nModel Configuration:")
    print(f"  Embedding dim:     {config.embed_dim}")
    print(f"  Attention heads:   {config.num_heads}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  Sequence length:   {seq_len}")

    # Create model
    model = GPTModel(config).to(device)
    num_params = model.count_parameters()
    print(f"  Parameters:        {num_params:,} ({num_params / 1_000_000:.1f}M)")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        text=text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.9,
        stride=seq_len,  # Non-overlapping sequences
    )

    print(f"\nData:")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size:         {batch_size}")

    # ---- Optimizer ----
    # AdamW: Adam optimizer with decoupled weight decay
    # - Adam tracks running averages of gradients (momentum)
    #   and squared gradients (adaptive learning rates per parameter)
    # - Weight decay adds a penalty for large weights (regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    print(f"\nTraining:")
    print(f"  Epochs:            {num_epochs}")
    print(f"  Learning rate:     {learning_rate}")
    print(f"  Optimizer:         AdamW (weight_decay=0.01)")

    # ---- Training Loop ----
    print("\n" + "-" * 60)
    print("TRAINING START")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()  # Enable dropout
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        # Progress bar for this epoch
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=True,
        )

        for input_ids, target_ids in progress_bar:
            # Move data to device (GPU/CPU)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass: compute predictions and loss
            _, loss = model(input_ids, targets=target_ids)

            # Backward pass: compute gradients
            # This is where backpropagation happens - PyTorch automatically
            # computes how each weight contributed to the loss
            loss.backward()

            # Gradient clipping: prevent exploding gradients
            # If gradients are too large, scale them down
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights using gradients
            optimizer.step()

            # Clear gradients for next batch
            # (gradients accumulate by default in PyTorch)
            optimizer.zero_grad()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- Epoch Summary ----
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        elapsed = time.time() - start_time

        # Evaluate on validation set
        if epoch % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)

            print(
                f"  Epoch {epoch:3d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, config, optimizer, epoch, save_dir, "best")
                print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
        else:
            print(
                f"  Epoch {epoch:3d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    # ---- Save Final Model ----
    save_checkpoint(model, config, optimizer, num_epochs, save_dir, "final")

    print("\n" + "-" * 60)
    print("TRAINING COMPLETE")
    print("-" * 60)
    print(f"\n  Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Best val loss:    {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {save_dir}/")

    return model, train_losses, val_losses


def save_checkpoint(
    model: GPTModel,
    config: GPTConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str,
    name: str,
):
    """
    Save a model checkpoint.

    Saves both the model weights and training state so training
    can be resumed later.
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        },
    }

    filepath = os.path.join(save_dir, f"gpt_{name}.pt")
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    device: torch.device | None = None,
) -> tuple[GPTModel, GPTConfig]:
    """
    Load a model from a checkpoint.

    Args:
        filepath: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        Tuple of (model, config)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    config = GPTConfig(**checkpoint["config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config


if __name__ == "__main__":
    train()
