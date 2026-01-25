"""
Embeddings Module for LLM From Scratch

Embeddings convert discrete token IDs into continuous vector representations
that neural networks can process and learn from.

Key Concepts:
- Token Embedding: Maps each vocabulary token to a learnable vector
- Positional Embedding: Encodes the position of each token in the sequence
- The final embedding is the sum of token + positional embeddings

PyTorch Concepts Introduced:
- nn.Module: Base class for all neural network components
- nn.Embedding: Lookup table that maps indices to vectors
- Tensors: Multi-dimensional arrays (like numpy arrays but with GPU support)
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors.

    This is essentially a lookup table where each token ID maps to a
    learnable vector. During training, these vectors are adjusted to
    capture semantic meaning.

    Example:
        vocab_size=50257, embed_dim=768
        Token ID 15496 ("Hello") → vector of 768 numbers

    PyTorch Notes:
        - nn.Module: Base class we inherit from. Handles parameter tracking,
          saving/loading, moving to GPU, etc.
        - nn.Embedding: Creates a weight matrix of shape (vocab_size, embed_dim).
          Given an index, it returns the corresponding row.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: Number of tokens in vocabulary (e.g., 50257 for GPT-2)
            embed_dim: Dimension of embedding vectors (e.g., 768)
        """
        # Always call parent __init__ first - this registers the module
        super().__init__()

        # nn.Embedding creates a matrix of shape (vocab_size, embed_dim)
        # Each row is a learnable vector for one token
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given token IDs.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)

        PyTorch Notes:
            - forward() defines what happens when you call the module
            - You call it like: output = module(input), not module.forward(input)
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Adds positional information to token embeddings.

    Transformers process all tokens in parallel, so they have no inherent
    sense of order. Positional embeddings tell the model where each token
    is in the sequence.

    We use learned positional embeddings (like GPT-2), where each position
    0, 1, 2, ... max_seq_len has its own learnable vector.

    Example:
        max_seq_len=1024, embed_dim=768
        Position 0 → vector of 768 numbers
        Position 1 → different vector of 768 numbers
        ...
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Args:
            max_seq_len: Maximum sequence length the model can handle (e.g., 1024)
            embed_dim: Dimension of embedding vectors (must match token embeddings)
        """
        super().__init__()

        # Create embeddings for positions 0 to max_seq_len-1
        self.embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate positional embeddings for a sequence.

        Args:
            seq_len: Length of the current sequence

        Returns:
            Tensor of shape (seq_len, embed_dim)

        PyTorch Notes:
            - torch.arange(n): Creates tensor [0, 1, 2, ..., n-1]
            - .to(device): Moves tensor to same device as model weights
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.embedding.weight.device)

        return self.embedding(positions)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer combining token and positional embeddings.

    This is what the GPT model will actually use. It:
    1. Converts token IDs to token embeddings
    2. Adds positional embeddings
    3. Applies dropout for regularization

    The output is ready to be fed into the transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embedding vectors
            max_seq_len: Maximum sequence length
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(max_seq_len, embed_dim)

        # Dropout randomly zeroes some elements during training
        # This prevents overfitting by forcing the model to not rely
        # too heavily on any single feature
        self.dropout = nn.Dropout(dropout)

        # Store dimensions for reference
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional information.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)

        Process:
            1. Look up token embeddings: (batch, seq_len) → (batch, seq_len, embed_dim)
            2. Look up position embeddings: seq_len → (seq_len, embed_dim)
            3. Add them together (broadcasting handles the batch dimension)
            4. Apply dropout
        """
        batch_size, seq_len = token_ids.shape

        # Get token embeddings: (batch_size, seq_len, embed_dim)
        token_embeds = self.token_embedding(token_ids)

        # Get positional embeddings: (seq_len, embed_dim)
        pos_embeds = self.position_embedding(seq_len)

        # Add them together
        # pos_embeds broadcasts across the batch dimension
        embeddings = token_embeds + pos_embeds

        # Apply dropout and return
        return self.dropout(embeddings)


def demo():
    """
    Demonstrate embedding functionality.

    Run this file directly to see embeddings in action:
        python -m src.embeddings
    """
    print("=" * 60)
    print("EMBEDDINGS DEMO")
    print("=" * 60)

    # Import our tokenizer
    from .tokenizer import Tokenizer

    # Configuration (small values for demo)
    vocab_size = 50257  # GPT-2 vocabulary size
    embed_dim = 768     # Standard GPT-2 embedding dimension
    max_seq_len = 1024  # Maximum sequence length
    dropout = 0.1

    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Dropout: {dropout}")

    # Create the embedding layer
    embedding_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # Count parameters
    num_params = sum(p.numel() for p in embedding_layer.parameters())
    print(f"\nTotal embedding parameters: {num_params:,}")
    print(f"  Token embedding: {vocab_size * embed_dim:,} ({vocab_size} × {embed_dim})")
    print(f"  Position embedding: {max_seq_len * embed_dim:,} ({max_seq_len} × {embed_dim})")

    print("\n" + "-" * 60)
    print("EMBEDDING EXAMPLE")
    print("-" * 60)

    # Tokenize some text
    tokenizer = Tokenizer()
    text = "Hello, I am learning to build an LLM!"
    token_ids = tokenizer.encode(text)

    print(f"\nText: {text!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")

    # Convert to PyTorch tensor and add batch dimension
    # Shape: (1, seq_len) - batch size of 1
    token_tensor = torch.tensor([token_ids])
    print(f"\nInput tensor shape: {token_tensor.shape}")
    print(f"  (batch_size=1, seq_len={len(token_ids)})")

    # Get embeddings (set to eval mode to disable dropout for demo)
    embedding_layer.eval()
    with torch.no_grad():  # Disable gradient computation for inference
        embeddings = embedding_layer(token_tensor)

    print(f"\nOutput embedding shape: {embeddings.shape}")
    print(f"  (batch_size=1, seq_len={len(token_ids)}, embed_dim={embed_dim})")

    # Show a slice of the embedding for the first token
    print(f"\nFirst token ('Hello') embedding (first 10 values):")
    print(f"  {embeddings[0, 0, :10].tolist()}")

    print("\n" + "-" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("-" * 60)

    # Show that we can process multiple sequences at once
    texts = [
        "Hello world",
        "How are you",
        "I am fine!",
    ]

    # Tokenize all texts
    all_tokens = [tokenizer.encode(t) for t in texts]

    # Pad to same length (simple padding for demo)
    max_len = max(len(t) for t in all_tokens)
    padded = [t + [tokenizer.eos_token_id] * (max_len - len(t)) for t in all_tokens]

    # Create batch tensor
    batch_tensor = torch.tensor(padded)
    print(f"\nBatch of {len(texts)} sequences:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text!r} → {all_tokens[i]}")

    print(f"\nBatch tensor shape: {batch_tensor.shape}")
    print(f"  (batch_size={len(texts)}, seq_len={max_len})")

    # Get embeddings for the batch
    with torch.no_grad():
        batch_embeddings = embedding_layer(batch_tensor)

    print(f"\nOutput shape: {batch_embeddings.shape}")
    print(f"  (batch_size={len(texts)}, seq_len={max_len}, embed_dim={embed_dim})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
