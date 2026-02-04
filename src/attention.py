"""
Multi-Head Attention Module for LLM From Scratch

Attention is the core mechanism of the transformer architecture.
It allows each token to "attend to" (look at) other tokens in the
sequence, learning contextual relationships between words.

Key Concepts:
- Query (Q): "What am I looking for?" - each token's search vector
- Key (K): "What do I contain?" - each token's label vector
- Value (V): "What info do I provide?" - each token's content vector
- Attention Score: How relevant two tokens are (dot product of Q and K)
- Causal Mask: Prevents tokens from looking at future tokens (GPT-style)
- Multi-Head: Multiple attention patterns learned in parallel

PyTorch Concepts Introduced:
- nn.Linear: Fully connected layer (matrix multiplication + bias)
- torch.matmul: Matrix multiplication
- torch.softmax: Converts scores to probabilities (0-1, sum to 1)
- torch.tril: Creates lower triangular matrix (used for causal masking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    "Causal" means each token can only attend to tokens at the same or
    earlier positions. This is essential for autoregressive (left-to-right)
    text generation - the model shouldn't see future tokens when predicting.

    "Self" means queries, keys, and values all come from the same sequence
    (as opposed to cross-attention where they come from different sequences).

    How it works (step by step):
        1. Project input into Q, K, V using learned linear layers
        2. Split into multiple heads
        3. Compute attention scores: score = Q · K^T / sqrt(d_k)
        4. Apply causal mask (hide future tokens)
        5. Apply softmax to get attention weights (probabilities)
        6. Multiply weights by V to get output
        7. Concatenate all heads and project back

    Example with 2 heads, embed_dim=768:
        - Each head works with 768/2 = 384 dimensions
        - Head 1 might learn syntactic relationships
        - Head 2 might learn semantic relationships
        - Their outputs are concatenated back to 768 dimensions
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Dimension of embeddings (e.g., 768). Must be divisible by num_heads
            num_heads: Number of attention heads (e.g., 12)
            max_seq_len: Maximum sequence length for causal mask
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        # Validate that embed_dim is evenly divisible by num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # nn.Linear(in_features, out_features) does: output = input @ W^T + b
        # We project from embed_dim to embed_dim, but will reshape into heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection: combines all heads back together
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # This ensures token at position i can only attend to positions 0..i
        #
        # For seq_len=4, the mask looks like:
        # [[1, 0, 0, 0],    Token 0 can only see token 0
        #  [1, 1, 0, 0],    Token 1 can see tokens 0-1
        #  [1, 1, 1, 0],    Token 2 can see tokens 0-2
        #  [1, 1, 1, 1]]    Token 3 can see tokens 0-3
        #
        # register_buffer: Saves this tensor with the model but it's NOT
        # a learnable parameter (no gradients). It moves with the model
        # to GPU/CPU automatically.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Step 1: Project input into Q, K, V
        # Each: (batch_size, seq_len, embed_dim)
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Step 2: Reshape into multiple heads
        # (batch, seq_len, embed_dim) → (batch, seq_len, num_heads, head_dim)
        # Then transpose to: (batch, num_heads, seq_len, head_dim)
        # This groups each head's data together for parallel computation
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention scores
        # Q @ K^T → (batch, num_heads, seq_len, seq_len)
        # Scale by sqrt(head_dim) to prevent softmax from saturating
        # (large dot products → extreme softmax values → vanishing gradients)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 4: Apply causal mask
        # Where mask is 0, fill attention scores with -infinity
        # After softmax, -inf becomes 0 (no attention to future tokens)
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )

        # Step 5: Apply softmax to get attention weights (probabilities)
        # Each row sums to 1.0, representing how much attention to pay
        # to each other token
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: Multiply attention weights by values
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim)
        # → (batch, heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Step 7: Concatenate heads
        # (batch, heads, seq_len, head_dim) → (batch, seq_len, heads, head_dim)
        # → (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Step 8: Final linear projection
        output = self.output_proj(attn_output)
        output = self.output_dropout(output)

        return output


def demo():
    """
    Demonstrate multi-head attention functionality.

    Run this file directly to see attention in action:
        python -m src.attention
    """
    print("=" * 60)
    print("MULTI-HEAD ATTENTION DEMO")
    print("=" * 60)

    from .tokenizer import Tokenizer
    from .embeddings import TransformerEmbedding

    # Configuration
    embed_dim = 768
    num_heads = 12
    max_seq_len = 1024
    dropout = 0.0  # Disable dropout for demo

    print(f"\nConfiguration:")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {embed_dim // num_heads}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create layers
    embedding = TransformerEmbedding(
        vocab_size=50257,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    attention = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # Count parameters
    attn_params = sum(p.numel() for p in attention.parameters())
    print(f"\nAttention parameters: {attn_params:,}")
    print(f"  Q projection: {embed_dim * embed_dim + embed_dim:,}")
    print(f"  K projection: {embed_dim * embed_dim + embed_dim:,}")
    print(f"  V projection: {embed_dim * embed_dim + embed_dim:,}")
    print(f"  Output projection: {embed_dim * embed_dim + embed_dim:,}")

    print("\n" + "-" * 60)
    print("ATTENTION FLOW")
    print("-" * 60)

    # Tokenize text
    tokenizer = Tokenizer()
    text = "The cat sat on the mat"
    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor([tokens])

    print(f"\nInput text: {text!r}")
    print(f"Tokens: {tokens}")
    print(f"Token strings: {[tokenizer.get_token_string(t) for t in tokens]}")

    # Run through embedding and attention
    embedding.eval()
    attention.eval()

    with torch.no_grad():
        # Step 1: Embed tokens
        embedded = embedding(token_tensor)
        print(f"\nAfter embedding: {embedded.shape}")
        print(f"  (batch=1, seq_len={len(tokens)}, embed_dim={embed_dim})")

        # Step 2: Apply attention
        attended = attention(embedded)
        print(f"\nAfter attention: {attended.shape}")
        print(f"  (batch=1, seq_len={len(tokens)}, embed_dim={embed_dim})")
        print(f"  Shape is preserved! Attention transforms values, not dimensions.")

    print("\n" + "-" * 60)
    print("CAUSAL MASK VISUALIZATION")
    print("-" * 60)

    # Show the causal mask for our sequence
    token_strs = [tokenizer.get_token_string(t) for t in tokens]
    seq_len = len(tokens)
    mask = attention.mask[0, 0, :seq_len, :seq_len]

    print(f"\nCausal mask ({seq_len}x{seq_len}):")
    print(f"1 = can attend, 0 = cannot attend (future token)\n")

    # Header
    header = "           " + "".join(f"{s:>8}" for s in token_strs)
    print(header)
    print("          " + "-" * (8 * seq_len))

    for i, token_str in enumerate(token_strs):
        row = f"{token_str:>8} | "
        for j in range(seq_len):
            val = int(mask[i, j].item())
            row += f"{'  ✓    ' if val else '  ·    '}"
        print(row)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
