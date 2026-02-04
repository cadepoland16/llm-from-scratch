"""
Dataset Module for LLM From Scratch

Handles loading text data and creating training batches.

The core idea:
- Load text and tokenize it into one long sequence of token IDs
- Create sliding windows of fixed length for training
- Each window becomes an (input, target) pair where target = input shifted by 1

Example with seq_len=4:
    Full text tokens: [10, 20, 30, 40, 50, 60, 70, 80]

    Window 1: input=[10, 20, 30, 40]  target=[20, 30, 40, 50]
    Window 2: input=[20, 30, 40, 50]  target=[30, 40, 50, 60]
    Window 3: input=[30, 40, 50, 60]  target=[40, 50, 60, 70]
    ...

PyTorch Concepts Introduced:
- torch.utils.data.Dataset: Base class for custom datasets
- torch.utils.data.DataLoader: Handles batching, shuffling, parallel loading
"""

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Dataset that creates overlapping sequences from tokenized text.

    PyTorch's Dataset class requires two methods:
    - __len__: Returns the total number of samples
    - __getitem__: Returns a single sample by index

    The DataLoader then uses these to create batches automatically.
    """

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        seq_len: int,
        stride: int | None = None,
    ):
        """
        Args:
            text: Raw text string to use as training data
            tokenizer: Tokenizer instance for encoding text
            seq_len: Length of each training sequence
            stride: Step size between sequences. If None, defaults to seq_len
                   (non-overlapping). Smaller stride = more samples but more overlap.
        """
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Tokenize the entire text at once
        self.tokens = tokenizer.encode(text)

        # Validate we have enough data
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"Text is too short ({len(self.tokens)} tokens) for "
                f"seq_len={seq_len}. Need at least {seq_len + 1} tokens."
            )

    def __len__(self) -> int:
        """
        Number of possible sequences.

        We need seq_len + 1 tokens per sample (input + 1 shifted target).
        """
        return max(0, (len(self.tokens) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) pair.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, target_ids), each of shape (seq_len,)
            target is input shifted by 1 position
        """
        start = idx * self.stride
        end = start + self.seq_len

        # Input: tokens[start:end]
        # Target: tokens[start+1:end+1] (shifted by 1)
        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        target_ids = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)

        return input_ids, target_ids


def create_dataloaders(
    text: str,
    tokenizer: Tokenizer,
    seq_len: int,
    batch_size: int,
    train_split: float = 0.9,
    stride: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from text.

    Splits the text into train/val portions, creates datasets,
    and wraps them in DataLoaders for batched iteration.

    Args:
        text: Raw text string
        tokenizer: Tokenizer instance
        seq_len: Sequence length for each sample
        batch_size: Number of samples per batch
        train_split: Fraction of data for training (default 0.9)
        stride: Step size between sequences

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Split text into train and validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, seq_len, stride)
    val_dataset = TextDataset(val_text, tokenizer, seq_len, stride)

    # Create DataLoaders
    # - shuffle=True for training (randomize order each epoch)
    # - shuffle=False for validation (consistent evaluation)
    # - drop_last=True to avoid incomplete batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader
