# Attention Variants: MHA, MQA, GQA, MLA

## The Problem: KV Cache Memory at Inference
During autoregressive generation, the KV cache stores keys and values for all previous tokens. For MHA with h heads:
- KV cache size = 2 × n_layers × seq_len × h × d_head × bytes_per_element
- For Llama-2 70B: ~140 GB for a 4K context — impractical for large batches

## Multi-Head Attention (MHA) — Original
- h query heads, h key heads, h value heads
- Each head has d_head = d_model / h dimensions
- Full expressivity but highest memory cost for KV cache

## Multi-Query Attention (MQA) — (Shazeer, 2019)
- h query heads but **only 1 key head and 1 value head** shared across all queries
- KV cache is h× smaller
- Significant quality degradation on some benchmarks
- Used in: PaLM, Falcon, early versions of code models

## Grouped-Query Attention (GQA) — (Ainslie et al., 2023)
- h query heads divided into g groups
- Each group shares 1 key head and 1 value head
- g key/value heads total (1 < g < h)
- **Interpolates between MHA (g=h) and MQA (g=1)**
- Almost no quality loss vs MHA with substantial memory savings
- Used in: **Llama-2 70B, Llama-3, Mistral 7B, Gemma, Mixtral**

### GQA Math:
  - Query heads per group: h/g
  - KV cache reduction factor: g (h/g times smaller than MHA)
  - Typical choice: g = 8 (e.g., 32 query heads → 8 KV heads)

## Multi-head Latent Attention (MLA) — (DeepSeek, 2024)
Introduced in DeepSeek-V2. Instead of caching full K,V:
1. Compress K,V into a **low-rank latent vector c_KV** of dimension d_c (much smaller than d_kv×h)
2. At inference, decompress c_KV back to K,V via learned up-projection matrices
3. Only cache c_KV — drastically smaller KV cache

**Key formula:**
  c_KV = W_DKV × h_t  (down-projection, d_c << n_h × d_h)
  K = W_UK × c_KV     (up-projection for keys)
  V = W_UV × c_KV     (up-projection for values)

**Benefits:**
- KV cache ≈ d_c per token (vs n_h × d_h per token for MHA)
- DeepSeek-V2 67B uses d_c = 512 vs MHA equivalent 8192 → 16× cache reduction
- Maintains near-MHA quality through the low-rank bottleneck

**Decoupled RoPE in MLA:** RoPE is applied to a separate query/key component, not the compressed representation, to preserve positional encoding fidelity.

## Comparison Table

| Variant | KV Heads | Cache Size | Quality | Used In |
|---------|----------|------------|---------|---------|
| MHA     | h        | 1×         | Best    | GPT-2, BERT |
| MQA     | 1        | 1/h        | Lower   | PaLM, Falcon |
| GQA     | g (1<g<h)| 1/g        | ≈MHA    | Llama-3, Mistral |
| MLA     | latent   | ~1/16      | ≈MHA    | DeepSeek-V2/V3 |
