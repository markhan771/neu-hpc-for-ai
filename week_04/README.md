# Week 04: FlashAttention-2 Implementation

Implementation of **Algorithm 1** (Section 3.1) from the [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) in both unparallelized C and parallelized CUDA.

## 📋 Assignment Requirements

### Part 1: Unparallelized C Implementation ✅
Implement Algorithm 1 using single-threaded C code to demonstrate understanding of the tiled computation and online softmax algorithm.

**File**: [flash_attention.c](flash_attention.c)

### Part 2: Parallelized CUDA Implementation ✅
Implement Algorithm 1 using CUDA with parallelization across row blocks.

**File**: [flash_attention.cu](flash_attention.cu)

### Goal
Achieve a **correct, parallelized implementation of FlashAttention** with focus on correctness over low-level optimizations (tensor cores, memory coalescing, etc. will be addressed later).

## 🎯 What is FlashAttention?

FlashAttention is an **IO-aware** attention algorithm that makes attention computation faster and more memory-efficient by:

1. **Tiling the computation** to fit in fast SRAM (shared memory)
2. **Using online softmax** to avoid materializing the full N×N attention matrix
3. **Reducing HBM (global memory) accesses** from O(N²d) to O(N²)

### Standard Attention vs FlashAttention

**Standard Attention** (memory inefficient):
```python
S = Q @ K^T / √d          # N×N matrix (huge!)
P = softmax(S)            # N×N matrix (huge!)
O = P @ V                 # Final output
```
- Memory: O(N²) - stores full attention matrix
- HBM accesses: O(N²d) reads + writes

**FlashAttention** (memory efficient):
```python
# Never materializes full N×N matrix!
# Processes in blocks that fit in SRAM
for each row block:
    for each column block:
        # Compute partial attention
        # Update running statistics
O = final normalized output
```
- Memory: O(Nd) - only stores Q, K, V, O
- HBM accesses: O(N²d / M) where M is SRAM size

## 📐 Algorithm 1: Forward Pass (Section 3.1)

### Inputs
- **Q, K, V**: Query, Key, Value matrices (N × d each)
- **N**: Sequence length
- **d**: Head dimension
- **Br, Bc**: Row and column block sizes

### Outputs
- **O**: Output matrix (N × d)
- **L**: Log-sum-exp vector (N) - saved for backward pass

### Key Variables (Online Softmax Statistics)

| Variable | Shape | Description |
|----------|-------|-------------|
| **m_i** | (Br,) | Running maximum per row (for numerical stability) |
| **l_i** | (Br,) | Running sum of exponentials (softmax denominator) |
| **acc** | (Br × d) | Accumulated (unnormalized) output |
| **S** | (Br × Bc) | Attention scores for current block |
| **P** | (Br × Bc) | Softmax probabilities for current block |

### Algorithm Pseudocode

```
For each row block i (rows [i*Br, (i+1)*Br)):
    Initialize:
        m_i = [-∞, -∞, ..., -∞]    # Br elements
        l_i = [0, 0, ..., 0]        # Br elements
        acc = zeros(Br, d)

    For each column block j (cols [j*Bc, (j+1)*Bc)):
        # 1. Compute attention scores
        S = Q_i @ K_j^T / √d        # (Br × Bc)

        # 2. Update running maximum
        m_ij = max(m_i, rowmax(S))  # Element-wise max

        # 3. Compute probabilities with new maximum
        P = exp(S - m_ij)           # (Br × Bc)

        # 4. Compute sum of probabilities
        l_ij = rowsum(P)            # (Br,)

        # 5. Compute rescaling factor
        alpha = exp(m_i - m_ij)     # (Br,)

        # 6. Rescale accumulator and add new contribution
        acc = acc * alpha + P @ V_j

        # 7. Update running sum
        l_i = l_i * alpha + l_ij

        # 8. Update running maximum
        m_i = m_ij

    # 9. Final normalization
    O_i = acc / l_i
    L_i = m_i + log(l_i)  # Save for backward pass
```

### Why This Works (Online Softmax)

The algorithm incrementally computes softmax without storing the full attention matrix:

**Standard softmax** (requires full row):
```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**Online softmax** (processes blocks):
1. Keep track of running max (`m_i`) and running sum (`l_i`)
2. When seeing new block:
   - Update max: `m_new = max(m_old, max_of_new_block)`
   - Rescale old contributions: `old_stuff *= exp(m_old - m_new)`
   - Add new contributions: `new_stuff = exp(new_block - m_new)`
   - Update sum: `l_new = l_old * exp(m_old - m_new) + sum(new_stuff)`

This ensures mathematically identical result to standard softmax!

## 📁 Project Structure

```
week_04/
├── flash_attention.c           # Part 1: Unparallelized C implementation
├── flash_attention.cu          # Part 2: Parallelized CUDA implementation
├── Makefile                    # Build system
├── modal_flash_attention.py    # Run on cloud GPU (Modal)
├── FLASHATTENTION.md           # Detailed algorithm explanation
└── README.md                   # This file
```

## 🔧 Building and Running

### Prerequisites

**For C version:**
- GCC or any C compiler
- No GPU required

**For CUDA version:**
- NVIDIA CUDA Toolkit (11.0+)
- NVIDIA GPU

### Local Build

```bash
cd week_04

# Build both versions
make

# Or build individually
make flash_attention_c      # C version only
make flash_attention_cuda   # CUDA version only

# Run C version
./flash_attention_c

# Run CUDA version (requires NVIDIA GPU)
./flash_attention_cuda

# Clean
make clean
```

### Run on Cloud GPU (Modal)

If you don't have a local GPU:

```bash
# Install Modal
pip install modal
modal setup

# Run CUDA version on cloud
modal run modal_flash_attention.py
```

## 🧪 Testing and Verification

Both implementations include:

1. **Reference Implementation**: Standard attention computation
   ```c
   O = softmax(Q @ K^T / √d) @ V
   ```

2. **Correctness Check**: Compare FlashAttention output vs reference
   - **Max error**: Maximum absolute difference
   - **Average error**: Mean absolute difference
   - **Pass threshold**: Max error < 1e-4 (C) or < 1e-3 (CUDA)

### Expected Output

```
FlashAttention-2 Forward Pass Test
===================================

Configuration:
  Sequence length (N): 128
  Head dimension (d):  64
  Row block size (Br): 32
  Col block size (Bc): 32

Running FlashAttention forward pass...
Done.

Running reference implementation...
Done.

Verifying correctness...
  Max error: 3.456e-05
  Avg error: 1.234e-06

[PASS] Test PASSED!
```

## 💻 Implementation Details

### Part 1: C Implementation ([flash_attention.c](flash_attention.c))

**Characteristics:**
- Single-threaded sequential execution
- Clear step-by-step implementation matching algorithm pseudocode
- Uses heap-allocated temporary arrays for each block
- Easier to understand and debug

**Key Features:**
- Outer loop over row blocks (Tr iterations)
- Inner loop over column blocks (Tc iterations)
- Explicit online softmax statistics tracking
- Clean separation of algorithm steps

**Test Configuration:**
```c
N = 128    // Sequence length
d = 64     // Head dimension
Br = 32    // Row block size
Bc = 32    // Column block size
```

### Part 2: CUDA Implementation ([flash_attention.cu](flash_attention.cu))

**Parallelization Strategy:**

1. **Grid Level**: One thread block per row tile
   ```
   Grid dimensions: (Tr, 1, 1) where Tr = ceil(N / Br)
   ```

2. **Block Level**: Threads cooperate within a block
   ```
   Block dimensions: (256, 1, 1)
   Processes Br rows together
   ```

3. **Thread Level**: Each thread processes multiple elements
   - Matrix multiplication: Strided access pattern
   - Reductions: Per-thread partial results

**Memory Hierarchy:**

```
Global Memory (HBM):
├── Q, K, V (input)     [Read-only, multiple reads per block]
├── O (output)          [Write-once per element]
└── L (logsumexp)       [Write-once per row]

Shared Memory (SRAM) per block:
├── m_i[Br]            [Running max]
├── l_i[Br]            [Running sum]
├── acc[Br × d]        [Accumulated output]
├── S[Br × Bc]         [Attention scores]
└── P[Br × Bc]         [Probabilities]

Registers:
└── Thread-local variables for computation
```

**Shared Memory Size:**
```c
size_t shared_mem = (Br + Br + Br*d + Br*Bc + Br*Bc) * sizeof(float)

For Br=64, Bc=64, d=64:
= (64 + 64 + 64*64 + 64*64 + 64*64) * 4 bytes
= (128 + 12288) * 4 = 49,664 bytes ≈ 48.5 KB
```

**Synchronization Points:**
```cuda
__syncthreads();  // After computing S
__syncthreads();  // After updating statistics
__syncthreads();  // After computing P @ V
```

**Test Configuration:**
```c
N = 512    // Larger sequence for GPU
d = 64     // Head dimension
Br = 64    // Larger blocks for GPU
Bc = 64
```

## 📊 Performance Characteristics

### Memory Complexity

| Implementation | Memory | HBM Reads | HBM Writes |
|---------------|---------|-----------|------------|
| Standard Attention | O(N² + Nd) | O(N²d) | O(N²d) |
| FlashAttention (Ours) | O(Nd) | O(N²d²/M) | O(Nd) |

Where M is SRAM size (48-192 KB per SM on modern GPUs).

### Computational Complexity

Both implementations have the same FLOPs: **O(N²d)**

FlashAttention does **more FLOPs** (recomputes attention scores in backward pass) but saves on memory IO, resulting in net speedup.

### Expected Speedup (vs Standard Attention)

- **Memory-bound regime** (large N, small d): ~3-10x faster
- **Compute-bound regime** (small N, large d): ~1-2x faster

**Note**: This implementation focuses on **correctness**, not performance. Production FlashAttention uses additional optimizations:
- Tensor cores for matrix multiplications
- Warp-level primitives
- Memory coalescing
- Register tiling
- Software pipelining

## 🔍 Key Implementation Challenges

### 1. Online Softmax Rescaling

**Challenge**: Update statistics when seeing new data

**Solution**: Rescale previous contributions by `exp(m_old - m_new)`

```c
float alpha = expf(m_i[r] - m_ij);
acc[r * d + k] *= alpha;
l_i[r] = l_i[r] * alpha + l_ij;
m_i[r] = m_ij;
```

### 2. Numerical Stability

**Challenge**: Prevent overflow in `exp()`

**Solution**: Subtract row maximum before exponentiation

```c
float m_ij = fmaxf(m_i[r], rowmax(S));
P[r * Bc + c] = expf(S[r * Bc + c] - m_ij);
```

### 3. Thread Cooperation (CUDA)

**Challenge**: Coordinate threads to compute shared results

**Solution**: Strided loops + synchronization

```cuda
// Each thread processes subset of elements
for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
    // Compute element idx
}
__syncthreads();  // Wait for all threads
```

### 4. Shared Memory Management

**Challenge**: Fit working set in limited shared memory

**Solution**: Careful memory layout, reuse buffers when possible

```c
// Allocate shared memory efficiently
float *m_i = shared_mem;
float *l_i = m_i + Br;
float *acc = l_i + Br;
float *S = acc + Br * d;
float *P = S + Br * Bc;  // Reuse for different blocks
```

## 📚 References

- **Paper**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
  - Authors: Tri Dao
  - Section 3.1: Algorithm 1 (Forward Pass)

- **Explanations**:
  - [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
  - [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
  - [Basic Idea Behind Flash Attention](https://damek.github.io/random/basic-idea-behind-flash-attention/)

- **Original FlashAttention (v1)**: [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

## ✅ Assignment Checklist

- [x] **Part 1**: Unparallelized C implementation
  - [x] Algorithm 1 correctly implemented
  - [x] Online softmax with proper rescaling
  - [x] Tiled computation over row and column blocks
  - [x] Reference implementation for verification
  - [x] Correctness tests passing

- [x] **Part 2**: Parallelized CUDA implementation
  - [x] Parallelization across row blocks
  - [x] Shared memory usage for temporary data
  - [x] Thread cooperation within blocks
  - [x] Proper synchronization
  - [x] Correctness tests passing

- [x] **Documentation**
  - [x] Algorithm explanation
  - [x] Implementation details
  - [x] Build instructions
  - [x] Testing procedure

## 🚀 Future Optimizations (Not Required for This Assignment)

These optimizations can achieve 5-10x additional speedup:

1. **Tensor Cores**: Use WMMA or CUTLASS for matrix multiplications
2. **Memory Coalescing**: Ensure 128-byte aligned, contiguous accesses
3. **Warp-Level Primitives**: Use shuffle instructions for reductions
4. **Register Tiling**: Keep frequently-accessed data in registers
5. **Software Pipelining**: Overlap computation and memory transfers
6. **Multi-Stage Processing**: Process multiple blocks simultaneously
7. **Attention Variants**: Support causal masking, different head dimensions

---

**Course**: NEU INFO7335 - HPC for AI
**Assignment**: Week 04 - FlashAttention-2 Implementation
**Date**: February 2026
**Focus**: Correctness and understanding of tiled attention computation
