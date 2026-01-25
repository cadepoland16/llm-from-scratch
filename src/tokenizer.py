"""
Tokenizer Module for LLM From Scratch

A tokenizer converts human-readable text into numerical tokens that the model
can process. We use Byte Pair Encoding (BPE) via OpenAI's tiktoken library,
which is the same tokenizer used by GPT models.

Key Concepts:
- Token: A subword unit (could be a word, part of a word, or character)
- Vocabulary: The complete set of all possible tokens
- Encoding: Converting text → list of token IDs
- Decoding: Converting token IDs → text
"""

import tiktoken


class Tokenizer:
    """
    BPE Tokenizer wrapper using tiktoken.

    Uses the GPT-2 tokenizer which has a vocabulary of 50,257 tokens.
    This includes:
    - Common words and subwords
    - Individual characters
    - Special token: <|endoftext|> (ID: 50256)

    Example:
        >>> tokenizer = Tokenizer()
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(tokens)
        [15496, 11, 995, 0]
        >>> text = tokenizer.decode(tokens)
        >>> print(text)
        Hello, world!
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the tokenizer.

        Args:
            model_name: The tiktoken encoding to use. Default is "gpt2".
                       Options: "gpt2", "r50k_base", "p50k_base", "cl100k_base"
        """
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.model_name = model_name

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        For GPT-2: 50,257 tokens
        This number is important because it determines the size of our
        embedding layer (each token needs its own embedding vector).
        """
        return self.tokenizer.n_vocab

    @property
    def eos_token_id(self) -> int:
        """
        Returns the End-Of-Sequence token ID.

        For GPT-2: 50256 (the <|endoftext|> token)
        This token marks the end of a text sequence and is used
        to separate different documents during training.
        """
        return self.tokenizer.eot_token

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """
        Convert text to a list of token IDs.

        Args:
            text: The input text to tokenize
            add_eos: Whether to append the end-of-sequence token

        Returns:
            List of integer token IDs

        Example:
            >>> tokenizer.encode("Hello")
            [15496]
        """
        token_ids = self.tokenizer.encode(text)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of integer token IDs

        Returns:
            Decoded text string

        Example:
            >>> tokenizer.decode([15496])
            'Hello'
        """
        return self.tokenizer.decode(token_ids)

    def get_token_string(self, token_id: int) -> str:
        """
        Get the string representation of a single token.

        Useful for debugging and understanding what each token represents.

        Args:
            token_id: A single token ID

        Returns:
            The string this token represents
        """
        return self.tokenizer.decode([token_id])


def demo():
    """
    Demonstrate tokenizer functionality.

    Run this file directly to see the tokenizer in action:
        python -m src.tokenizer
    """
    print("=" * 60)
    print("TOKENIZER DEMO")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = Tokenizer()
    print(f"\nTokenizer: {tokenizer.model_name}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"End-of-sequence token ID: {tokenizer.eos_token_id}")

    # Example texts to tokenize
    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!",
        "GPT stands for Generative Pre-trained Transformer.",
    ]

    print("\n" + "-" * 60)
    print("ENCODING EXAMPLES")
    print("-" * 60)

    for text in examples:
        tokens = tokenizer.encode(text)
        print(f"\nText: {text!r}")
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")

        # Show what each token represents
        token_strings = [tokenizer.get_token_string(t) for t in tokens]
        print(f"Token breakdown: {token_strings}")

    print("\n" + "-" * 60)
    print("DECODING TEST")
    print("-" * 60)

    # Verify encoding/decoding is reversible
    original = "Building an LLM from scratch!"
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {original!r}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded!r}")
    print(f"Match: {original == decoded}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
