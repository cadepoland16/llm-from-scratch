"""
Text Generation Module for LLM From Scratch

This module implements autoregressive text generation: given a prompt,
the model predicts one token at a time, appending each prediction to
the input for the next step.

Sampling Strategies:
- Greedy: Always pick the most likely token (deterministic)
- Temperature: Control randomness (higher = more creative, lower = more focused)
- Top-k: Only consider the k most likely tokens
- Top-p (nucleus): Only consider tokens with cumulative probability ≤ p

These strategies control the trade-off between coherence and creativity.

Key Concept:
    The model outputs "logits" - raw scores for every token in the vocabulary.
    We convert logits → probabilities → sampled token using these strategies.
"""

import torch
import torch.nn.functional as F

from .tokenizer import Tokenizer
from .gpt_model import GPTModel
from .train import load_checkpoint, get_device


@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Generate text given a prompt.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (0.0 = greedy, 1.0 = normal, >1.0 = more random)
        top_k: If set, only sample from the top k most likely tokens
        top_p: If set, only sample from tokens with cumulative probability ≤ p
        device: Device to run on

    Returns:
        Generated text (including the original prompt)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Truncate to max sequence length if needed
        input_tokens = tokens[:, -model.config.max_seq_len:]

        # Forward pass to get logits
        logits, _ = model(input_tokens)

        # We only care about the last position's predictions
        # (what comes next after the entire sequence so far)
        next_logits = logits[:, -1, :]  # (batch=1, vocab_size)

        # Apply temperature
        # Higher temperature → flatter distribution → more random
        # Lower temperature → peakier distribution → more deterministic
        # Temperature of 0 would be greedy (handled separately)
        if temperature == 0.0:
            # Greedy: pick the single most likely token
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                next_logits = top_k_filter(next_logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                next_logits = top_p_filter(next_logits, top_p)

            # Convert logits to probabilities
            probs = F.softmax(next_logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to our sequence
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop if we generate an end-of-sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode all tokens back to text
    generated_ids = tokens[0].tolist()
    return tokenizer.decode(generated_ids)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top k highest logits, set the rest to -infinity.

    This prevents the model from choosing very unlikely tokens.

    Example with k=3 and vocab of 5:
        logits:   [2.0, 5.0, 1.0, 4.0, 3.0]
        filtered: [-inf, 5.0, -inf, 4.0, 3.0]  (only top 3 kept)

    After softmax, -inf becomes 0 probability.
    """
    if k <= 0:
        return logits

    # Find the kth largest value
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)  # kth largest value

    # Set everything below the kth value to -inf
    return logits.masked_fill(logits < min_top_k, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose
    cumulative probability ≥ p.

    This adapts the number of tokens considered based on the
    distribution - if the model is very confident, fewer tokens
    are considered. If uncertain, more tokens are included.

    Example with p=0.9:
        If token A has 95% probability → only A is considered
        If top 5 tokens each have ~20% → all 5 are considered
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find tokens to remove (cumulative prob exceeds p)
    # Shift right so the token that pushes over p is kept
    sorted_mask = cumulative_probs - sorted_probs > p

    # Set filtered logits to -inf
    sorted_logits[sorted_mask] = float("-inf")

    # Restore original order
    original_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return original_logits


def demo():
    """
    Demonstrate text generation with different sampling strategies.

    Run this file directly:
        python -m src.generate
    """
    import os

    print("=" * 60)
    print("TEXT GENERATION DEMO")
    print("=" * 60)

    # Load trained model
    checkpoint_path = os.path.join("models", "gpt_final.pt")
    if not os.path.exists(checkpoint_path):
        print(f"\nNo trained model found at {checkpoint_path}")
        print("Please run training first: python -m src.train")
        return

    device = get_device()
    model, config = load_checkpoint(checkpoint_path, device)
    model.eval()
    tokenizer = Tokenizer()

    print(f"\nModel loaded from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Parameters: {model.count_parameters():,}")

    # Different prompts to try
    prompts = [
        "The",
        "I had",
        "It was",
    ]

    # Strategy 1: Greedy (temperature=0)
    print("\n" + "-" * 60)
    print("STRATEGY 1: GREEDY (temperature=0)")
    print("Always picks the most likely token. Deterministic output.")
    print("-" * 60)

    for prompt in prompts:
        text = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {text!r}")

    # Strategy 2: Temperature sampling
    print("\n" + "-" * 60)
    print("STRATEGY 2: TEMPERATURE SAMPLING")
    print("Higher temp = more random, lower = more focused.")
    print("-" * 60)

    prompt = "The"
    for temp in [0.5, 1.0, 1.5]:
        text = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=temp)
        print(f"\nTemp={temp}: {text!r}")

    # Strategy 3: Top-k sampling
    print("\n" + "-" * 60)
    print("STRATEGY 3: TOP-K SAMPLING")
    print("Only consider the k most likely tokens.")
    print("-" * 60)

    prompt = "I had"
    for k in [5, 20, 50]:
        text = generate(
            model, tokenizer, prompt, max_new_tokens=50,
            temperature=0.8, top_k=k,
        )
        print(f"\nTop-k={k}: {text!r}")

    # Strategy 4: Top-p (nucleus) sampling
    print("\n" + "-" * 60)
    print("STRATEGY 4: TOP-P (NUCLEUS) SAMPLING")
    print("Adaptively consider tokens until cumulative prob >= p.")
    print("-" * 60)

    prompt = "It was"
    for p in [0.5, 0.9, 0.95]:
        text = generate(
            model, tokenizer, prompt, max_new_tokens=50,
            temperature=0.8, top_p=p,
        )
        print(f"\nTop-p={p}: {text!r}")

    print("\n" + "=" * 60)
    print("NOTE: This model was trained on a small text (~5K tokens).")
    print("Larger datasets and more training will produce better output!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
