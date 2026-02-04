"""
GPT Model for LLM From Scratch

This module brings together all components into a complete GPT model:
- Tokenizer: text → token IDs
- Embeddings: token IDs → vectors (with positional info)
- Transformer Blocks: process and refine representations
- Output Head: vectors → vocabulary logits (next-token predictions)

The model is autoregressive: it predicts one token at a time,
using all previous tokens as context.

Key Concepts:
- Logits: Raw (unnormalized) scores for each vocabulary token.
          Higher logit = model thinks that token is more likely next.
- LM Head: The final linear layer that projects from embed_dim to vocab_size
- Weight Tying: Sharing weights between token embeddings and output layer

PyTorch Concepts Introduced:
- nn.ModuleList: A list of modules properly registered with PyTorch
- dataclass: Python's built-in way to define configuration objects
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from .embeddings import TransformerEmbedding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.

    Using a dataclass keeps all hyperparameters organized in one place.
    This makes it easy to experiment with different model sizes.

    The defaults below match a small GPT-2 style model (~124M parameters).
    """
    vocab_size: int = 50257      # GPT-2 vocabulary size
    max_seq_len: int = 1024      # Maximum context window
    embed_dim: int = 768         # Embedding dimension
    num_heads: int = 12          # Number of attention heads
    num_layers: int = 12         # Number of transformer blocks
    dropout: float = 0.1        # Dropout probability

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )


class GPTModel(nn.Module):
    """
    Complete GPT Language Model.

    Architecture:
        1. TransformerEmbedding: token IDs → embedded vectors
        2. TransformerBlock × num_layers: process through attention + FFN
        3. LayerNorm: final normalization
        4. Linear (LM Head): project to vocabulary logits

    The model predicts the probability distribution over the vocabulary
    for the NEXT token at each position.

    For input "The cat sat":
        Position 0 ("The")  → predicts what comes after "The"
        Position 1 ("cat")  → predicts what comes after "The cat"
        Position 2 ("sat")  → predicts what comes after "The cat sat"
    """

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig object with model hyperparameters
        """
        super().__init__()

        self.config = config

        # Token + positional embeddings
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Stack of transformer blocks
        # nn.ModuleList ensures PyTorch properly tracks all parameters
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.embed_dim)

        # Language model head: projects from embed_dim to vocab_size
        # Output is a score (logit) for every token in the vocabulary
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output layer
        # This is a key technique used in GPT-2 and many other models.
        # Intuition: if "cat" has embedding vector [0.2, -0.1, ...], then
        # when the model wants to predict "cat", the output should also
        # align with that same vector. Sharing weights enforces this.
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights.

        Good initialization is crucial for training stability:
        - Linear layers: small random normal values (std=0.02)
        - Embeddings: small random normal values (std=0.02)
        - Biases: zeros
        - LayerNorm: weight=1, bias=0 (identity transform initially)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for computing loss, shape (batch_size, seq_len)
                    If None, loss is not computed (inference mode)

        Returns:
            Tuple of (logits, loss):
            - logits: shape (batch_size, seq_len, vocab_size)
            - loss: scalar tensor if targets provided, else None
        """
        # Step 1: Embed tokens (adds positional information)
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(token_ids)

        # Step 2: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 3: Final layer normalization
        x = self.ln_final(x)

        # Step 4: Project to vocabulary logits
        # (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # Step 5: Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Cross-entropy loss expects:
            #   input: (N, C) where C is number of classes (vocab_size)
            #   target: (N,) with class indices
            # So we flatten: (batch * seq_len, vocab_size) and (batch * seq_len,)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Preset configurations for different model sizes
GPT_CONFIGS = {
    "small": GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
    ),
    "medium": GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        dropout=0.1,
    ),
    "large": GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=1280,
        num_heads=20,
        num_layers=36,
        dropout=0.1,
    ),
}


def demo():
    """
    Demonstrate the complete GPT model.

    Run this file directly:
        python -m src.gpt_model
    """
    print("=" * 60)
    print("GPT MODEL DEMO")
    print("=" * 60)

    from .tokenizer import Tokenizer

    # Use a smaller config for the demo to keep it fast
    config = GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.0,
    )

    print(f"\nModel Configuration:")
    print(f"  Vocabulary size:   {config.vocab_size:,}")
    print(f"  Max sequence len:  {config.max_seq_len}")
    print(f"  Embedding dim:     {config.embed_dim}")
    print(f"  Attention heads:   {config.num_heads}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  Head dimension:    {config.embed_dim // config.num_heads}")

    # Create model
    model = GPTModel(config)
    model.eval()

    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1_000_000:.1f}M)")

    # Parameter breakdown
    embed_params = sum(p.numel() for p in model.embedding.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    ln_params = sum(p.numel() for p in model.ln_final.parameters())
    head_params = model.config.vocab_size * model.config.embed_dim  # weight tied

    print(f"\nParameter Breakdown:")
    print(f"  Embeddings:         {embed_params:,}")
    print(f"  Transformer blocks: {block_params:,}")
    print(f"  Final LayerNorm:    {ln_params:,}")
    print(f"  LM Head:            (weight tied with embeddings)")

    print("\n" + "-" * 60)
    print("FORWARD PASS")
    print("-" * 60)

    # Tokenize input
    tokenizer = Tokenizer()
    text = "The future of artificial intelligence is"
    tokens = tokenizer.encode(text)
    token_tensor = torch.tensor([tokens])

    print(f"\nInput: {text!r}")
    print(f"Tokens: {[tokenizer.get_token_string(t) for t in tokens]}")
    print(f"Token IDs: {tokens}")
    print(f"Input shape: {token_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        logits, loss = model(token_tensor)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  (batch=1, seq_len={len(tokens)}, vocab_size={config.vocab_size})")

    # Show top predictions for the last token
    # The last position's logits predict what comes NEXT
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = torch.softmax(last_logits, dim=-1)
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nTop {top_k} predicted next tokens after '{text}':")
    print(f"  (Note: model is untrained, so predictions are random)")
    for i in range(top_k):
        token_str = tokenizer.get_token_string(top_indices[i].item())
        prob = top_probs[i].item() * 100
        print(f"  {i+1}. {token_str!r:15} ({prob:.2f}%)")

    print("\n" + "-" * 60)
    print("LOSS COMPUTATION")
    print("-" * 60)

    # Show loss computation with targets
    # For training, targets are the input shifted by 1:
    # Input:  "The future of  AI  is"
    # Target: "future of  AI  is  [next]"
    input_ids = token_tensor[:, :-1]   # All tokens except last
    target_ids = token_tensor[:, 1:]   # All tokens except first

    print(f"\nInput tokens:  {[tokenizer.get_token_string(t) for t in input_ids[0].tolist()]}")
    print(f"Target tokens: {[tokenizer.get_token_string(t) for t in target_ids[0].tolist()]}")

    with torch.no_grad():
        logits, loss = model(input_ids, targets=target_ids)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Expected loss for random predictions: {torch.log(torch.tensor(config.vocab_size * 1.0)).item():.4f}")
    print(f"  (These should be close since the model is untrained)")

    print("\n" + "-" * 60)
    print("MODEL SIZE COMPARISON")
    print("-" * 60)

    print(f"\nPreset configurations:")
    for name, cfg in GPT_CONFIGS.items():
        m = GPTModel(cfg)
        params = m.count_parameters()
        print(f"  GPT-{name:6s}: {params / 1_000_000:>7.1f}M params "
              f"({cfg.num_layers} layers, {cfg.embed_dim} dim, {cfg.num_heads} heads)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
