# Week 02: CUDA GEMM Implementation

This assignment implements a naive CUDA GEMM (Generalized Matrix Multiplication) kernel using only global memory, without high-level libraries like cuBLAS or cuDNN.

## 📋 Assignment Overview

### Part 1: Naive GEMM Kernel (Required)

**Task**: Implement a naive GEMM kernel that computes:

$$D = \alpha \cdot AB + \beta C$$

Where:
- $A$ is an $m \times k$ matrix
- $B$ is a $k \times n$ matrix
- $C$ is an $m \times n$ matrix
- $D$ is an $m \times n$ matrix
- $\alpha$, $\beta$ are scalar coefficients

**Implementation**: See [`gemm.cu:30-45`](gemm.cu#L30-L45)

```cuda
__global__ void gemm_naive_out(
    int m, int n, int k,
    float alpha, const float* A, const float* B,
    float beta,  const float* C, float* D
)
```

**Key characteristics:**
- Uses only global memory (no shared memory or tiling)
- Each thread computes one element of the output matrix $D$
- 2D grid/block configuration: `dim3 block(16,16)` and `dim3 grid(ceil_div(n,16), ceil_div(m,16))`
- Thread $(x,y)$ computes element $D[row][col]$ where `row = blockIdx.y * blockDim.y + threadIdx.y` and `col = blockIdx.x * blockDim.x + threadIdx.x`

### Part 2: Extended GEMM Kernel (Required)

**Task**: Extend the GEMM kernel to:
1. Optionally transpose either $A$ or $B$
2. Update $C$ in-place instead of allocating a separate $D$

The kernel should support these operations:
- $C \leftarrow \alpha \cdot A^{T}B + \beta \cdot C$
- $C \leftarrow \alpha \cdot AB^{T} + \beta \cdot C$
- $C \leftarrow \alpha \cdot A^{T}B^{T} + \beta \cdot C$
- $C \leftarrow \alpha \cdot AB + \beta \cdot C$

**Implementation**: See [`gemm.cu:78-94`](gemm.cu#L78-L94)

```cuda
__global__ void gemm_naive_inplace_trans(
    int m, int n, int k,
    float alpha, const float* A, const float* B,
    float beta, float* C,
    int transA, int transB
)
```

**Key design decisions:**

1. **Transpose convention** (see [`gemm.cu:47-58`](gemm.cu#L47-L58)):
   - When `transA=0`: $A$ is stored as $(m \times k)$, operation uses $A$ directly
   - When `transA=1`: $A$ is stored as $(k \times m)$, operation uses $A^T$ (which is $m \times k$)
   - Same convention for $B$

2. **Loading functions** (see [`gemm.cu:60-76`](gemm.cu#L60-L76)):
   ```cuda
   __device__ __forceinline__ float loadA(
       const float* A, int m, int k, int row, int t, int transA)

   __device__ __forceinline__ float loadB(
       const float* B, int k, int n, int t, int col, int transB)
   ```
   - These helper functions handle the memory access pattern based on transpose flags
   - For `transA=0`: accesses `A[row*k + t]` (row-major)
   - For `transA=1`: accesses `A[t*m + row]` (accessing transposed layout)

3. **In-place update**:
   - Reads original `C[row*n + col]` value
   - Computes new value: `alpha * sum + beta * oldc`
   - Writes back to the same location in `C`

## 🏗️ Project Structure

```
week_02/
├── gemm.cu          # Complete CUDA implementation with both kernels
├── Makefile         # Build automation
└── README.md        # This file
```

## 🔧 Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (tested with CUDA 11.0+)
- NVIDIA GPU with compute capability 7.0+ (adjust `-arch` flag in Makefile if needed)

### Compilation

```bash
# Build the executable
make

# Or compile manually
nvcc -O2 -arch=sm_70 gemm.cu -o gemm
```

### Running Tests

```bash
# Run the program
make run

# Or run directly
./gemm
```

### Expected Output

```
Using GPU: [Your GPU Name]
[PASS] naive GEMM (writes D) tests passed.
[PASS] extended GEMM (transpose + in-place C) tests passed ✅
All done.
```

## 🧪 Testing Strategy

The implementation includes comprehensive testing:

### 1. Corner Cases (see [`gemm.cu:282-291`](gemm.cu#L282-L291))
- **Minimal size**: $1 \times 1 \times 1$ matrices
- **Vector results**: $1 \times 5 \times 1$
- **Rectangular**: $2 \times 3 \times 1$
- **Small square**: $2 \times 2 \times 2$
- **Medium**: $3 \times 4 \times 5$
- **Non-multiples of block size**: $17 \times 19 \times 23$
- **Larger squares**: $64 \times 64 \times 64$, $128 \times 96 \times 80$

### 2. Transpose Combinations (see [`gemm.cu:299-306`](gemm.cu#L299-L306))
For each test case, all four transpose combinations are tested:
- `transA=0, transB=0`: $C \leftarrow \alpha \cdot AB + \beta C$
- `transA=1, transB=0`: $C \leftarrow \alpha \cdot A^T B + \beta C$
- `transA=0, transB=1`: $C \leftarrow \alpha \cdot A B^T + \beta C$
- `transA=1, transB=1`: $C \leftarrow \alpha \cdot A^T B^T + \beta C$

### 3. Randomized Tests (see [`gemm.cu:308-316`](gemm.cu#L308-L316))
- 50 random test cases with dimensions $m, n, k \in [1, 64]$
- Random transpose flags
- Tests edge cases not covered by manual test cases

### 4. Correctness Verification

Each test compares GPU results against CPU reference implementation:
- CPU reference: See [`gemm.cu:109-126`](gemm.cu#L109-L126)
- Comparison uses `allclose()` with tolerances: `atol=1e-4`, `rtol=1e-3` (see [`gemm.cu:136-144`](gemm.cu#L136-L144))
- Tests fail immediately on first mismatch with diagnostic output

## 💡 Implementation Details

### Memory Layout

Matrices are stored in row-major order (C convention):
- Element $(i, j)$ of an $m \times n$ matrix is at index $i \times n + j$
- For transposed storage, dimensions are swapped

### Thread Organization

```
Block size: 16 × 16 threads
Grid size:  ceil(n/16) × ceil(m/16) blocks

Each thread computes one output element:
  row = blockIdx.y * 16 + threadIdx.y
  col = blockIdx.x * 16 + threadIdx.x
```

### Algorithm (Naive Approach)

For each output element $C[i][j]$:
1. Initialize `sum = 0`
2. Loop over $k$: `sum += op(A)[i][t] * op(B)[t][j]`
3. Compute final value: `C[i][j] = alpha * sum + beta * C_old[i][j]`

**Time Complexity**: $O(mnk)$ operations
**Memory Access Pattern**: Global memory only (naive, no optimization)

### Why This is "Naive"

This implementation is deliberately simple and not optimized:
- ❌ No shared memory usage
- ❌ No memory coalescing optimization
- ❌ No tiling/blocking strategies
- ❌ No register blocking
- ✅ Only global memory reads/writes

For production code, you would use:
- Shared memory tiling (10-100x faster)
- Memory coalescing patterns
- cuBLAS library (highly optimized, 100-1000x faster)

## 📊 Performance Characteristics

### Expected Performance
- Small matrices ($< 100$): Dominated by kernel launch overhead
- Medium matrices ($100$-$1000$): Shows GPU advantage but far from peak
- Large matrices ($> 1000$): Better utilization but still limited by memory bandwidth

### Limitations
1. **Memory bandwidth bound**: Each output element requires $k$ reads from $A$ and $k$ reads from $B$
2. **No data reuse**: Unlike tiled implementations, this kernel doesn't exploit data locality
3. **Uncoalesced access**: Especially for transposed matrices, memory access patterns are not optimal

## 🎓 Learning Objectives

This assignment demonstrates:
1. ✅ Basic CUDA kernel structure and launch configuration
2. ✅ 2D grid/block organization for matrix operations
3. ✅ Global memory access patterns
4. ✅ Handling matrix transposes in CUDA
5. ✅ In-place updates in CUDA kernels
6. ✅ Testing and verification strategies
7. ✅ Understanding why naive implementations are slow (motivation for optimization)

## 📝 Code Organization

- **Lines 30-45**: Part 1 - Basic GEMM kernel (`gemm_naive_out`)
- **Lines 60-76**: Helper functions for transpose-aware memory access
- **Lines 78-94**: Part 2 - Extended GEMM kernel (`gemm_naive_inplace_trans`)
- **Lines 102-126**: CPU reference implementation for verification
- **Lines 131-144**: Helper functions for random initialization and comparison
- **Lines 170-219**: Test harness for extended kernel
- **Lines 224-269**: Test harness for basic kernel
- **Lines 271-321**: Main function with comprehensive test suite

## 🚀 Next Steps (Not Required)

To optimize this further (beyond assignment scope):
1. Implement shared memory tiling
2. Optimize memory coalescing
3. Use register blocking
4. Add prefetching
5. Implement warp-level optimizations
6. Compare with cuBLAS performance

## 📚 References

- CUDA Programming Guide: [Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- cuBLAS GEMM: [cublas<t>gemm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)

---

**Course**: NEU INFO7335 - HPC for AI
**Assignment**: Week 02 - CUDA GEMM Implementation
**Date**: February 2026
