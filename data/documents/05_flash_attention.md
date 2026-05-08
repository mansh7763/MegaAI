# Flash Attention

## The Problem: Memory Bandwidth Bottleneck
Standard attention materialises the full n×n attention matrix in HBM (GPU high-bandwidth memory):
- Writing S = QK^T/√d_k to HBM: O(n²) memory reads/writes
- For n=4096, d_model=1024: ~67 MB for the attention matrix alone
- GPU compute has far outpaced memory bandwidth (A100: 312 TFLOPS FP16, 2 TB/s HBM)
- Standard attention is memory-bound, not compute-bound

## Flash Attention (Dao et al., 2022 — Stanford + ETH Zurich)
**Key insight**: Recompute attention on-the-fly using tiling instead of materialising the full matrix.

### Algorithm
1. Divide Q, K, V into tiles that fit in SRAM (L2/shared memory)
2. For each tile of Q: iterate over tiles of K and V in SRAM
3. Compute partial softmax + weighted sum within SRAM using the online softmax trick
4. Never write the full n×n matrix to HBM

### Online Softmax Trick
Computing softmax in a streaming fashion:
- Track running max m_i and normalisation sum l_i
- Update running statistics as each K tile is processed
- Final output is exact (no approximation) despite never seeing the full attention matrix

### Results (Flash Attention v1)
- 2-4× speedup in wall-clock time vs standard attention on A100
- 5-20× reduction in memory usage
- Enables training on sequences 5-10× longer

## Flash Attention 2 (Dao, 2023)
Improvements over FA1:
1. **Better work partitioning**: reduce non-matmul FLOPs, maximise GPU occupancy
2. **Parallelism over sequence length**: distribute work across thread blocks along the sequence dimension
3. **Causal masking optimisation**: skip lower-triangular blocks that are fully masked
- **3-5× faster than standard attention, 2× faster than FA1**
- Used in: Llama, Mistral, Falcon, all major open-source models

## Flash Attention 3 (2024)
Targets H100/H800 GPUs with:
- FP8 support via Hopper TMA and wgmma instructions
- Asynchronous pipelining of GEMM and softmax
- Achieves up to 75% theoretical FP8 FLOPS on H100

## Flash Decoding
Extension of FA for inference (not just training):
- During decode, Q has batch_size queries but K, V are very long
- Parallelise over the KV sequence dimension using partial softmax reduction
- Enables fast long-context inference with KV cache

## Impact
Flash Attention made the following practical:
- Long-context training (32K, 128K, 1M tokens)
- Large batch sizes fitting in GPU memory
- Efficient fine-tuning (LoRA + FA2 is the standard recipe)
