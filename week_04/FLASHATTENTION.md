# FlashAttention-2 Implementation (INFO7335)

This directory contains implementations of **Algorithm 1** from Section 3.1 of the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691).

## Overview

FlashAttention is an IO-aware implementation of attention that reduces memory reads/writes and speeds up attention computation by:

1. **Tiling**: Breaking the computation into blocks that fit in fast SRAM
2. **Online Softmax**: Computing softmax incrementally without materializing the full attention matrix
3. **Recomputation**: Trading off FLOPs for memory IO in the backward pass

This implementation focuses on the **forward pass** with correct tiled computation.

## Algorithm 1: Forward Pass

The algorithm computes attention `O = softmax(QK^T / √d)V` using tiled computation:

### Key Components:
- **Q, K, V**: Query, Key, Value matrices (N × d)
- **Br, Bc**: Row and column block sizes
- **Online statistics**:
  - `m_i`: Running maximum for numerical stability
  - `l_i`: Running sum of exponentials (softmax denominator)
  - `acc`: Accumulated output (unnormalized until the end)

### Algorithm Steps (per row block):
1. Initialize: `m_i = -∞`, `l_i = 0`, `acc = 0`
2. For each column block (K_j, V_j):
   - Compute attention scores: `S = Q_i @ K_j^T / √d`
   - Update max: `m_ij = max(m_i, rowmax(S))`
   - Compute probabilities: `P = exp(S - m_ij)`
   - Compute sum: `l_ij = rowsum(P)`
   - Rescale accumulator: `acc = acc * exp(m_i - m_ij) + P @ V_j`
   - Update statistics: `l_i = l_i * exp(m_i - m_ij) + l_ij`, `m_i = m_ij`
3. Final normalization: `O_i = acc / l_i`

## Files

### Implementations
- **[flash_attention.c](flash_attention.c)**
  Unparallelized C implementation with CPU reference for correctness verification

- **[flash_attention.cu](flash_attention.cu)**
  Parallelized CUDA implementation with per-row-block parallelization

### Build & Deploy
- **[Makefile](Makefile)**
  Build system for both C and CUDA versions

- **[modal_flash_attention.py](modal_flash_attention.py)**
  Script to compile and run CUDA code on Modal cloud GPUs

## Building and Running

### Local Build (requires GCC and CUDA toolkit)

```bash
# Build both implementations
make

# Build only C version
make flash_attention_c

# Build only CUDA version
make flash_attention_cuda

# Run tests
make test

# Clean build artifacts
make clean
```

### Run Locally

```bash
# C implementation
./flash_attention_c

# CUDA implementation (requires NVIDIA GPU)
./flash_attention_cuda
```

### Run on Modal (cloud GPU)

```bash
modal run modal_flash_attention.py
```

## Implementation Details

### C Implementation ([flash_attention.c](flash_attention.c))
- **Single-threaded** execution for clarity
- Processes one row block at a time sequentially
- Useful for understanding the algorithm
- Includes standard attention reference implementation for verification

**Configuration:**
- Sequence length: 128
- Head dimension: 64
- Block sizes: Br=32, Bc=32

### CUDA Implementation ([flash_attention.cu](flash_attention.cu))

**Parallelization Strategy:**
- One thread block per row tile (Tr blocks total)
- Each block processes Br rows independently
- Within each block, threads cooperate using shared memory
- No inter-block synchronization needed (embarrassingly parallel across rows)

**Memory Hierarchy:**
- **Global memory (HBM)**: Q, K, V, O, L
- **Shared memory (SRAM)**: Temporary S, P matrices and statistics (m_i, l_i, acc)
- **Registers**: Thread-local computations

**Configuration:**
- Sequence length: 512
- Head dimension: 64
- Block sizes: Br=64, Bc=64
- Threads per block: 256

**Kernel Design:**
```
Grid:  (Tr, 1, 1) where Tr = ceil(N / Br)
Block: (256, 1, 1)
Shared memory per block: (Br + Br + Br*d + Br*Bc + Br*Bc) * sizeof(float)
```

## Correctness Verification

Both implementations compare their output against a reference standard attention implementation:

```
Standard Attention: O = softmax(QK^T / √d) V
```

The test measures:
- **Max error**: Maximum absolute difference between outputs
- **Average error**: Mean absolute difference

Expected error: < 1e-4 for C, < 1e-3 for CUDA (due to different floating point operation orders)

## Performance Notes

This implementation prioritizes **correctness** over performance:
- ✅ Correct tiled computation with online softmax
- ✅ Parallelized across row blocks
- ⏸️ Not optimized (no tensor cores, memory coalescing, etc.)

Future optimizations could include:
- Shared memory tiling within blocks
- Warp-level optimizations
- Tensor core usage for matrix multiplications
- Memory coalescing for global memory access
- Block-level reduction optimizations

## References

- **Paper**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- **Section**: 3.1 (Algorithm 1 - Forward Pass)
- **Authors**: Tri Dao

## Testing Output Example

```
FlashAttention-2 CUDA Forward Pass Test
========================================

Configuration:
  Sequence length (N): 512
  Head dimension (d):  64
  Row block size (Br): 64
  Col block size (Bc): 64

Running FlashAttention CUDA kernel...
Done. Time: 2.345 ms

Running CPU reference implementation...
Done.

Verifying correctness...
  Max error:       3.456e-04
  Avg error:       1.234e-05
  Large errors:    0 / 32768

✓ Test PASSED!
```

## License

Educational implementation for INFO7335 coursework.
