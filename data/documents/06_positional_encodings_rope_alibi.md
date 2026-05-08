# Positional Encodings: RoPE, ALiBi, and Absolute

## Why Positional Encoding?
Self-attention is permutation-invariant — it produces the same output regardless of token order. Positional encodings inject position information into token representations.

## Absolute Sinusoidal (Original Transformer)
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

- Added to token embeddings before the first layer
- Fixed (not learned)
- Poor extrapolation beyond training length

## Learned Absolute Positional Embeddings
- Trainable embedding matrix of size [max_len, d_model]
- Used in BERT, GPT-2
- Cannot generalise beyond max_len seen during training

## Relative Positional Encoding (Shaw et al., 2018; T5)
- Encode relative distance between positions, not absolute position
- More robust to sequence length changes
- T5 uses a simplified version with learned buckets

## Rotary Positional Embeddings (RoPE) — (Su et al., 2021)
**Core idea**: rotate Q and K vectors in 2D subspaces by an angle proportional to position.

For position m, dimension pair (2i, 2i+1):
  q_m rotated by θ_i×m, where θ_i = 10000^(-2i/d_head)

**Key properties:**
- The inner product q_m · k_n depends only on the relative distance (m - n)
- Naturally encodes relative position through rotation geometry
- Extrapolates reasonably beyond training length with YaRN/NTK scaling
- **Used in: Llama, Mistral, Falcon, Gemma, Qwen, DeepSeek**

### RoPE Extension Techniques
- **NTK-aware scaling**: rescale base frequency to extend context
- **YaRN (Yet Another RoPE Extension)**: dynamic interpolation for 2-8× context extension
- **LongRoPE**: position interpolation preserving nearby positions accurately

## ALiBi (Attention with Linear Biases) — (Press et al., 2022)
Instead of adding to embeddings, ALiBi adds a linear bias to attention logits:
  attn_score(i,j) = q_i · k_j / √d_k - m × (i - j)

Where m is a per-head slope (fixed geometric sequence).

**Properties:**
- No positional embeddings in token representation
- Extrapolates beyond training length by design (bias naturally penalises longer distances)
- Used in: BLOOM, MPT

## Comparison

| Method   | Extrapolation | Relative | Learned | Used In |
|----------|--------------|----------|---------|---------|
| Absolute | Poor         | No       | Optional| BERT, GPT-2 |
| T5 Bias  | Moderate     | Yes      | Yes     | T5, FLAN |
| RoPE     | Good (w/ scaling) | Yes | No  | Llama, Mistral |
| ALiBi    | Best native  | Yes      | No      | BLOOM, MPT |
