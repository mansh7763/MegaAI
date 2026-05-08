# KV Cache

## What Is the KV Cache?
During autoregressive generation, the Transformer recomputes attention over all previous tokens at each step. The KV cache avoids this by storing the computed keys (K) and values (V) for all previous positions and reusing them.

**Without KV cache**: generating token t requires O(t²) total compute (recompute all previous K, V)
**With KV cache**: generating token t requires O(t) compute (only compute K, V for new token; reuse cache)

## How It Works
1. **Prefill phase**: process the entire prompt in parallel, compute and store K, V for all positions → fills the KV cache
2. **Decode phase**: generate one token at a time, compute K, V only for the new token, append to cache, then compute attention over full (cached + new) K, V

## Cache Size Formula
For a batch:
  cache_bytes = 2 × n_layers × n_heads × d_head × seq_len × batch_size × dtype_bytes

For Llama-3 70B (FP16):
  = 2 × 80 × 8 × 128 × seq_len × batch × 2 bytes
  = ~0.33 GB per 1K tokens per batch item

## KV Cache Challenges

### Memory Pressure
- Long contexts fill GPU VRAM quickly
- Batching is limited by KV cache memory, not compute
- A 100K context window with batch=8 requires ~26 GB just for KV cache on Llama-3 70B

### PagedAttention (vLLM)
PagedAttention allocates KV cache in non-contiguous pages (like OS virtual memory), enabling:
- Dynamic allocation — no need to pre-allocate max sequence length
- Sharing KV cache between requests with common prefixes (prefix caching)
- Enables 2-4× higher throughput vs naive KV cache management

### Sliding Window Attention (Mistral)
Only attend to the last W tokens instead of all previous. KV cache is bounded at W regardless of sequence length. Long-range information is propagated through layers.

### KV Cache Quantization
- Quantize cached K, V to INT8 or INT4
- Reduces memory by 2-4× with minimal quality loss
- Used in: llama.cpp, TensorRT-LLM, vLLM

## Prefix Caching
If many requests share a common prefix (e.g., a long system prompt), the KV cache for that prefix is computed once and reused across requests. This dramatically reduces Time-To-First-Token (TTFT) for shared-prefix workloads.

## Speculative Decoding and KV Cache
In speculative decoding, a draft model proposes k tokens, then the target model verifies all k in one forward pass. The KV cache must be rolled back to the last verified position if some tokens are rejected.
