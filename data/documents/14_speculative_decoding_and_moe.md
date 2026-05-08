# Speculative Decoding and Mixture of Experts

## Speculative Decoding

### The Problem
Autoregressive generation is sequential — each token requires one full forward pass. For large models (70B+), this is slow even with optimised kernels.

### How Speculative Decoding Works (Leviathan et al., 2022)
1. A small **draft model** (fast, cheap) generates k candidate tokens autoregressively
2. The large **target model** verifies all k tokens in **one parallel forward pass**
3. If token i is accepted (probability ratio check), keep it; if rejected, resample from target distribution starting at position i
4. Guaranteed to produce the same distribution as the target model

**Speedup**: if the draft model has high acceptance rate, k tokens per target model call → up to 3-4× speedup with no quality loss.

### Draft Model Choices
- A smaller version of the same model family (Llama-70B verifies Llama-7B drafts)
- n-gram based: use a lookup table of common continuations
- Medusa (Cai et al., 2024): multiple small draft heads on top of the target model itself

### Acceptance Rate
Acceptance rate depends on how well the draft model matches the target. For similar-distribution tasks (coding, chat), rates of 70-90% are typical.

---

## Mixture of Experts (MoE)

### What Is MoE?
Instead of one dense FFN layer, use N expert FFN layers. A learned **router** selects top-k experts for each token.

  FFN_output = Σ router_weight_i × Expert_i(x)   for top-k experts

### Sparse Activation
Only k of N experts activate for each token (typically k=2, N=8 or N=64). This means:
- Total parameters: N × d_ff × d_model (large)
- Active parameters per token: k × d_ff × d_model (small)
- Same inference compute as a much smaller dense model

### Models Using MoE
- **Mixtral 8×7B** (Mistral AI): 8 experts, top-2 routing, ~46B total params but ~12.9B active
- **Mixtral 8×22B**: 8 experts, ~141B total, ~39B active
- **GPT-4** (rumoured): 8 experts × ~220B params
- **DeepSeek-V3**: 256 experts, top-8 routing, 671B total params, 37B active

### Routing Strategies
- **Top-K routing**: select k experts with highest router logits
- **Expert choice routing**: each expert selects its top-c tokens (load balancing by design)
- **Auxiliary loss**: balance load across experts by penalising under/over-utilised experts

### Challenges
- **Load imbalance**: without auxiliary loss, some experts are never used ("expert collapse")
- **Communication overhead**: in distributed settings, all-to-all communication for expert routing
- **Memory**: all expert weights must fit in memory, even if only k activate per token

### MoE vs Dense Models
At the same inference compute (FLOPs), MoE models have:
- More total parameters (hence more knowledge)
- Lower per-token compute cost
- Higher memory footprint

Mixtral 8×7B matches or beats Llama-2 70B on most benchmarks while using ~6× less compute per token.
