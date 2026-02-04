"""
Transformer Block Module for LLM From Scratch

A transformer block is the fundamental repeating unit of GPT.
The full GPT model stacks multiple of these blocks on top of each other.

Each block contains:
1. Layer Normalization → Multi-Head Attention → Residual Connection
2. Layer Normalization → FeedForward Network → Residual Connection

Key Concepts:
- Layer Normalization: Normalizes activations to stabilize training
- FeedForward Network: A small MLP that processes each token independently
- Residual Connection: Adds the input back to the output (skip connection)
- GELU Activation: Smooth activation function used in GPT (alternative to ReLU)

Why this architecture works:
- Attention captures relationships BETWEEN tokens
- FeedForward adds processing capacity for EACH token
- Layer norm keeps values in a reasonable range
- Residual connections let gradients flow through deep networks

PyTorch Concepts Introduced:
- nn.LayerNorm: Normalizes across the feature dimension
- nn.GELU: Gaussian Error Linear Unit activation function
- nn.Sequential: Chains multiple layers together
"""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise FeedForward Network.

    A simple two-layer MLP applied to each token independently.
    This is where the model does most of its "thinking" - the attention
    layer gathers information, and the feedforward layer processes it.

    Architecture:
        Linear(embed_dim → 4 * embed_dim)  ← expand
        GELU activation                     ← non-linearity
        Linear(4 * embed_dim → embed_dim)  ← project back
        Dropout                            ← regularization

    The 4x expansion is standard in transformers. It gives the network
    more capacity in the intermediate representation.

    Why GELU instead of ReLU?
        ReLU:  f(x) = max(0, x)         — hard cutoff at 0
        GELU:  f(x) = x * Φ(x)         — smooth, probabilistic cutoff
        GELU works better in practice for language models.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of input/output (e.g., 768)
            dropout: Dropout probability
        """
        super().__init__()

        # nn.Sequential chains layers together automatically
        # Input flows through each layer in order
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # Expand: 768 → 3072
            nn.GELU(),                              # Activation
            nn.Linear(4 * embed_dim, embed_dim),   # Project back: 3072 → 768
            nn.Dropout(dropout),                    # Regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block combining attention and feedforward layers.

    This is the core building block of GPT. The full model stacks
    multiple of these blocks (e.g., 12 blocks in GPT-2 small).

    Architecture (GPT-2 "pre-norm" style):
        x → LayerNorm → Attention → + (add x back) → LayerNorm → FFN → + (add back) → output

    Why "pre-norm"?
        Original transformer: Attention → Add → LayerNorm (post-norm)
        GPT-2 style:          LayerNorm → Attention → Add (pre-norm)
        Pre-norm is more stable for training deep networks.

    Why residual connections?
        Without them: output = Attention(x)
        With them:    output = x + Attention(LayerNorm(x))

        The "x +" part means the original information can flow straight
        through. The attention just adds refinements. This prevents the
        "vanishing gradient" problem in deep networks - gradients can
        flow directly through the skip connections during backpropagation.
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
            embed_dim: Dimension of embeddings (e.g., 768)
            num_heads: Number of attention heads (e.g., 12)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        # Layer norms (one before attention, one before feedforward)
        # nn.LayerNorm normalizes across the last dimension (embed_dim)
        # This keeps values centered around 0 with unit variance
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Attention and feedforward layers
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.feedforward = FeedForward(embed_dim=embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Attention block with residual connection
        # x + Attention(LayerNorm(x))
        x = x + self.attention(self.ln1(x))

        # FeedForward block with residual connection
        # x + FFN(LayerNorm(x))
        x = x + self.feedforward(self.ln2(x))

        return x


def demo():
    """
    Demonstrate transformer block functionality.

    Run this file directly:
        python -m src.transformer_block
    """
    print("=" * 60)
    print("TRANSFORMER BLOCK DEMO")
    print("=" * 60)

    from .tokenizer import Tokenizer
    from .embeddings import TransformerEmbedding

    # Configuration
    embed_dim = 768
    num_heads = 12
    max_seq_len = 1024

    print(f"\nConfiguration:")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  FeedForward inner dim: {4 * embed_dim}")
    print(f"  Max sequence length: {max_seq_len}")

    # Create layers
    embedding = TransformerEmbedding(
        vocab_size=50257,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )

    # Count parameters
    block_params = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for p in block.attention.parameters())
    ff_params = sum(p.numel() for p in block.feedforward.parameters())
    ln_params = sum(p.numel() for p in block.ln1.parameters()) + \
                sum(p.numel() for p in block.ln2.parameters())

    print(f"\nTransformer Block parameters: {block_params:,}")
    print(f"  Attention: {attn_params:,}")
    print(f"  FeedForward: {ff_params:,}")
    print(f"  Layer Norms: {ln_params:,}")

    print("\n" + "-" * 60)
    print("DATA FLOW THROUGH THE BLOCK")
    print("-" * 60)

    # Tokenize
    tokenizer = Tokenizer()
    text = "Transformers are the backbone of modern AI"
    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor([tokens])

    print(f"\nInput text: {text!r}")
    print(f"Tokens: {[tokenizer.get_token_string(t) for t in tokens]}")

    # Process
    embedding.eval()
    block.eval()

    with torch.no_grad():
        # Embed
        embedded = embedding(token_tensor)
        print(f"\n1. After embedding:       {embedded.shape}")

        # Through transformer block
        output = block(embedded)
        print(f"2. After transformer block: {output.shape}")

    print(f"\n   Shape preserved: {embedded.shape == output.shape}")

    print("\n" + "-" * 60)
    print("STACKING BLOCKS (PREVIEW)")
    print("-" * 60)

    # Show what happens with multiple blocks (like the full GPT model)
    num_blocks = 12
    blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, max_seq_len, dropout=0.0)
        for _ in range(num_blocks)
    ])

    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"\n  Single block params: {block_params:,}")
    print(f"  Stacked {num_blocks} blocks: {total_params:,}")
    print(f"  That's {total_params / 1_000_000:.1f}M parameters just in the blocks!")

    # Run through all blocks
    with torch.no_grad():
        x = embedded
        for i, b in enumerate(blocks):
            b.eval()
            x = b(x)
        print(f"\n  After {num_blocks} blocks: {x.shape}")
        print(f"  Shape still preserved through all {num_blocks} blocks!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
