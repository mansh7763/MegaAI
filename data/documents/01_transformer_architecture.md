# Transformer Architecture

## Origin
The Transformer was introduced in "Attention Is All You Need" (Vaswani et al., 2017, Google Brain). It replaced RNNs and CNNs for sequence modelling tasks, relying entirely on self-attention to compute representations of input and output.

## Core Architecture

### Encoder-Decoder Structure
The original Transformer has two stacks:
- **Encoder**: maps input sequence to continuous representations. Each encoder layer has: (1) multi-head self-attention, (2) position-wise feed-forward network. Both use residual connections + LayerNorm.
- **Decoder**: generates output autoregressively. Each decoder layer adds a cross-attention sub-layer that attends to encoder output.

### Encoder-only Models (BERT family)
- Processes full sequence bidirectionally
- Used for classification, NER, embeddings
- Examples: BERT, RoBERTa, DeBERTa, ELECTRA

### Decoder-only Models (GPT family)
- Causal (left-to-right) attention only
- Used for generation, instruction following
- Examples: GPT-2, GPT-3, GPT-4, Llama, Mistral, Gemma, Qwen

### Key Components
1. **Token Embeddings**: maps vocabulary indices to dense vectors of dimension d_model
2. **Positional Encoding**: adds position information since attention is permutation-invariant
3. **Multi-Head Attention**: h parallel attention heads with projections
4. **Feed-Forward Network**: two linear layers with GELU/ReLU, dimension d_ff (typically 4×d_model)
5. **Layer Normalization**: pre-norm (modern) or post-norm (original) placement
6. **Residual Connections**: x + Sublayer(x) to prevent vanishing gradients

## Scaling Laws
Chinchilla (Hoffmann et al., 2022) showed that optimal training computes tokens = 20 × parameters. GPT-3 (175B params) trained on 300B tokens — undertrained. Llama-2 (70B) trained on 2T tokens — more compute-optimal.

## Modern Variants
- **Prefix LM**: allows bidirectional attention on the prefix (T5, GLM)
- **Mixture of Experts (MoE)**: sparse activation of expert FFN layers (Mixtral, GPT-4 rumoured)
- **State Space Models (SSM)**: Mamba uses selective state spaces as an alternative to attention for linear-time sequence modelling
