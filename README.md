# LLM From Scratch

A complete PyTorch implementation of a GPT-style Large Language Model built entirely from the ground up. Every component — from tokenization to text generation — is implemented with thorough documentation explaining the underlying concepts.

## Project Overview

This project builds a transformer-based language model similar to GPT-2, implementing each component from scratch to deeply understand the architecture behind modern LLMs. The model is trained on sample text data and can generate new text using multiple sampling strategies.

### Key Highlights

- **16M parameter** GPT model (configurable up to 774M)
- Trained on Apple Silicon GPU (MPS) with PyTorch
- Four text generation strategies: greedy, temperature, top-k, and top-p sampling
- Every module is documented with explanations of both ML concepts and PyTorch patterns

## Architecture

The model follows the GPT-2 transformer architecture:

```
Input Text
    │
    ▼
┌─────────────────────┐
│  Tokenizer (BPE)    │  Text → Token IDs
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Token Embeddings   │  Token IDs → Vectors (768-dim)
│  + Position Embeds  │  Add positional information
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Transformer Block  │ ×N (12 layers in GPT-2 small)
│  ┌───────────────┐  │
│  │ Layer Norm    │  │
│  │ Multi-Head    │  │  12 attention heads
│  │ Attention     │  │  Causal masking
│  │ + Residual    │  │
│  ├───────────────┤  │
│  │ Layer Norm    │  │
│  │ FeedForward   │  │  768 → 3072 → 768
│  │ (MLP + GELU)  │  │
│  │ + Residual    │  │
│  └───────────────┘  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Layer Norm         │
│  Linear (LM Head)   │  Vectors → Vocabulary Logits
└─────────┬───────────┘
          ▼
    Next Token Prediction
```

### Model Configurations

| Config | Parameters | Layers | Embed Dim | Heads |
|--------|-----------|--------|-----------|-------|
| Training | 16.0M | 4 | 256 | 8 |
| GPT-2 Small | 124.4M | 12 | 768 | 12 |
| GPT-2 Medium | 354.8M | 24 | 1024 | 16 |
| GPT-2 Large | 774.0M | 36 | 1280 | 20 |

## Project Structure

```
llm-from-scratch/
├── src/
│   ├── __init__.py              # Package exports
│   ├── tokenizer.py             # BPE tokenizer using tiktoken
│   ├── embeddings.py            # Token + positional embeddings
│   ├── attention.py             # Multi-head causal self-attention
│   ├── transformer_block.py     # Transformer block (attention + FFN)
│   ├── gpt_model.py             # Complete GPT model + config
│   ├── dataset.py               # Text dataset and data loading
│   ├── train.py                 # Training pipeline
│   └── generate.py              # Text generation with sampling
├── data/                        # Training data (downloaded automatically)
├── models/                      # Saved model checkpoints
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cadepoland16/llm-from-scratch.git
cd llm-from-scratch
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on sample text data (downloads automatically):

```bash
python -m src.train
```

Training uses Apple Silicon GPU (MPS) or NVIDIA GPU (CUDA) automatically when available, falling back to CPU otherwise.

### Text Generation

Generate text using a trained model:

```bash
python -m src.generate
```

### Running Individual Components

Each module includes a demo that can be run independently:

```bash
python -m src.tokenizer           # Tokenizer demo
python -m src.embeddings          # Embeddings demo
python -m src.attention           # Attention demo with causal mask visualization
python -m src.transformer_block   # Transformer block demo
python -m src.gpt_model           # Full model demo with parameter breakdown
```

### Using as a Library

```python
from src import Tokenizer, GPTConfig, GPTModel, generate, load_checkpoint

# Load a trained model
model, config = load_checkpoint("models/gpt_final.pt")

# Generate text
tokenizer = Tokenizer()
output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
)
print(output)
```

## Component Deep Dive

### 1. Tokenizer (`src/tokenizer.py`)
Wraps OpenAI's tiktoken library for Byte Pair Encoding (BPE) tokenization. Converts text into subword tokens using GPT-2's 50,257-token vocabulary.

### 2. Embeddings (`src/embeddings.py`)
Converts token IDs into dense vectors and adds positional information. Uses learned embeddings for both tokens (50,257 x embed_dim) and positions (max_seq_len x embed_dim).

### 3. Multi-Head Attention (`src/attention.py`)
Implements scaled dot-product attention with causal masking. Splits the embedding into multiple heads that learn different relationship patterns in parallel.

### 4. Transformer Block (`src/transformer_block.py`)
Combines multi-head attention with a feedforward network (MLP), using pre-norm layer normalization and residual connections for stable deep training.

### 5. GPT Model (`src/gpt_model.py`)
Assembles all components into a complete language model. Includes weight tying between the token embedding and output projection layers, and configurable model sizes.

### 6. Training Pipeline (`src/train.py` + `src/dataset.py`)
Full training loop with AdamW optimizer, gradient clipping, validation evaluation, and model checkpointing. Dataset creates sliding-window (input, target) pairs for next-token prediction.

### 7. Text Generation (`src/generate.py`)
Autoregressive generation with four sampling strategies:
- **Greedy** (temperature=0): Deterministic, always picks the most likely token
- **Temperature**: Controls randomness — lower is more focused, higher is more creative
- **Top-k**: Limits sampling to the k most probable tokens
- **Top-p (Nucleus)**: Dynamically selects tokens until cumulative probability reaches p

## Future Improvements

If continuing development, the following enhancements would strengthen the project:

- **Larger training datasets** — Train on a larger corpus (e.g., OpenWebText, BookCorpus) to improve generation quality and generalization beyond the small sample text
- **Learning rate scheduling** — Implement warmup with cosine decay for smoother convergence and better final performance
- **Load pretrained GPT-2 weights** — Verify the architecture by loading OpenAI's original GPT-2 weights and comparing outputs
- **Distributed training** — Add multi-GPU support with PyTorch DistributedDataParallel for training larger model configurations
- **Evaluation metrics** — Implement perplexity scoring and standard NLP benchmarks to quantitatively measure model quality
- **Fine-tuning pipeline** — Add support for instruction fine-tuning and LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Jupyter notebook walkthrough** — Create an interactive notebook that walks through each component with visualizations
- **KV-cache optimization** — Implement key-value caching during generation to avoid redundant computation
- **Flash Attention** — Integrate memory-efficient attention for handling longer sequences
- **Quantization** — Add model quantization (INT8/INT4) for faster inference and reduced memory usage

## Technologies

- **Python 3.13** — Core language
- **PyTorch 2.10** — Deep learning framework
- **tiktoken** — BPE tokenization (OpenAI)
- **NumPy** — Numerical computing
- **tqdm** — Training progress bars
- **Matplotlib** — Visualization

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by Sebastian Raschka's *Build a Large Language Model (From Scratch)* and the original GPT-2 paper by OpenAI.
