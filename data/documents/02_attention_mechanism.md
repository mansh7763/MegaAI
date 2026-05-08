# Attention Mechanism

## Scaled Dot-Product Attention
The fundamental operation:

  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

Where:
- **Q** (Query): what we're looking for
- **K** (Key): what each position offers
- **V** (Value): the actual content to aggregate
- **d_k**: key dimension (used for scaling to prevent softmax saturation)

## Multi-Head Attention (MHA)
Runs h attention heads in parallel, each with its own learned projections W_Q, W_K, W_V:

  MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O
  head_i = Attention(QW_Q_i, KW_K_i, VW_V_i)

**Why multiple heads?** Each head can specialise on a different aspect: one head may track syntactic dependencies, another semantic relationships, another positional proximity.

**Complexity**: O(n²·d) time and O(n²) memory for sequence length n. This quadratic scaling is the core bottleneck of the Transformer.

## Self-Attention vs Cross-Attention
- **Self-attention**: Q, K, V all come from the same sequence (encoder or decoder attending to itself)
- **Cross-attention**: Q comes from decoder, K and V come from encoder outputs

## Types of Attention Masking
- **Causal (autoregressive) mask**: future positions are masked to −∞ before softmax, preventing look-ahead
- **Padding mask**: pad tokens are masked to prevent attending to meaningless positions
- **Prefix mask**: bidirectional on prefix, causal on generated continuation

## Attention in Modern LLMs
Modern decoder-only LLMs (GPT-4, Llama-3, Mistral) use:
- **Pre-normalization** (LayerNorm before attention, not after)
- **RoPE** (Rotary Positional Embeddings) instead of absolute sinusoidal
- **SwiGLU** activation in FFN layers instead of ReLU
- **GQA or MQA** instead of full MHA for inference efficiency
