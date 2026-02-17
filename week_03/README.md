# Week 03: CUDA Matrix Multiplication Optimization

Replication of all code runs and calculations from the worklog article: [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance**](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm.

## 📋 Assignment Overview

This assignment replicates a comprehensive study of CUDA matrix multiplication (SGEMM) optimizations, progressing from a naive implementation to achieving **93.7% of cuBLAS performance** through systematic optimizations.

### Original Article Results (NVIDIA A6000)

The article implements 12 different kernel versions with progressive optimizations:

| Kernel | Optimization Technique | GFLOPs/s | vs cuBLAS |
|--------|------------------------|----------|-----------|
| 0 | cuBLAS (baseline) | 23,249.6 | 100.0% |
| 1 | Naive | 309.0 | 1.3% |
| 2 | Global Memory Coalescing | 1,986.5 | 8.5% |
| 3 | Shared Memory Caching | 2,980.3 | 12.8% |
| 4 | 1D Blocktiling | 8,474.7 | 36.5% |
| 5 | 2D Blocktiling | 15,971.7 | 68.7% |
| 6 | Vectorized Memory Access | 18,237.3 | 78.4% |
| 7 | Avoid Bank Conflicts (Linearize) | 16,213.4 | 69.7% |
| 8 | Avoid Bank Conflicts (Offset) | 16,459.2 | 70.8% |
| 9 | Autotuning | 19,721.0 | 84.8% |
| 10 | Warptiling | 21,779.3 | 93.7% |
| 11 | Double Buffering | 17,278.3 | 74.3% |

## 🏗️ Project Structure

```
week_03/
├── SGEMM_CUDA/              # Cloned repository from siboehm/SGEMM_CUDA
│   ├── src/
│   │   └── kernels/
│   │       ├── 1_naive.cuh                    # Kernel 1: Basic implementation
│   │       ├── 2_kernel_global_mem_coalesce.cuh  # Kernel 2: Memory coalescing
│   │       ├── 3_kernel_shared_mem_blocking.cuh  # Kernel 3: Shared memory
│   │       ├── 4_kernel_1D_blocktiling.cuh    # Kernel 4: 1D tiling
│   │       ├── 5_kernel_2D_blocktiling.cuh    # Kernel 5: 2D tiling
│   │       ├── 6_kernel_vectorize.cuh         # Kernel 6: Vectorization
│   │       ├── 7_kernel_resolve_bank_conflicts.cuh  # Kernel 7
│   │       ├── 8_kernel_bank_extra_col.cuh    # Kernel 8
│   │       ├── 9_kernel_autotuned.cuh         # Kernel 9: Autotuned params
│   │       ├── 10_kernel_warptiling.cuh       # Kernel 10: Warp-level tiling
│   │       ├── 11_kernel_double_buffering.cuh # Kernel 11: Double buffering
│   │       └── 12_kernel_double_buffering.cuh # Kernel 12: Enhanced version
│   ├── sgemm.cu             # Main benchmark runner
│   ├── cuBLAS_sgemm.cu      # cuBLAS baseline
│   └── CMakeLists.txt       # Build configuration
├── run_on_h100.py           # Modal script for H100 execution
└── README.md                # This file
```

## 🚀 Running on H100 GPU

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal setup
   ```

### Execution

Run all kernels on H100:
```bash
cd week_03
modal run run_on_h100.py
```

Run a specific kernel:
```bash
# Run kernel 10 (Warptiling)
modal run run_on_h100.py --kernel 10
```

### What the Script Does

1. **System Setup**
   - Provisions H100 GPU instance
   - Installs CUDA 12.4 toolkit
   - Displays GPU information

2. **Build Configuration**
   - Clones the SGEMM_CUDA repository
   - Configures for H100 (compute capability 9.0)
   - Compiles all kernels with CMake

3. **Benchmark Execution**
   - Runs each kernel individually
   - Measures performance at matrix size 4096×4096
   - Collects GFLOPs/s metrics

4. **Results Collection**
   - Captures performance data
   - Generates comparison plots
   - Displays summary statistics

## 📊 H100 Results

> **Note**: Run the Modal script above to populate this section with actual H100 results

### GPU Information
```
[To be filled after running on H100]
- GPU: NVIDIA H100
- Compute Capability: 9.0
- CUDA Version: 12.4
- Memory: 80GB HBM3
```

### Performance Results

| Kernel | Optimization | H100 GFLOPs/s | vs cuBLAS | A6000 GFLOPs/s | Speedup |
|--------|--------------|---------------|-----------|----------------|---------|
| 0 | cuBLAS | [TBD] | 100.0% | 23,249.6 | [TBD] |
| 1 | Naive | [TBD] | [TBD] | 309.0 | [TBD] |
| 2 | GMEM Coalescing | [TBD] | [TBD] | 1,986.5 | [TBD] |
| 3 | SMEM Caching | [TBD] | [TBD] | 2,980.3 | [TBD] |
| 4 | 1D Blocktiling | [TBD] | [TBD] | 8,474.7 | [TBD] |
| 5 | 2D Blocktiling | [TBD] | [TBD] | 15,971.7 | [TBD] |
| 6 | Vectorized Access | [TBD] | [TBD] | 18,237.3 | [TBD] |
| 7 | Bank Conflicts (Linearize) | [TBD] | [TBD] | 16,213.4 | [TBD] |
| 8 | Bank Conflicts (Offset) | [TBD] | [TBD] | 16,459.2 | [TBD] |
| 9 | Autotuning | [TBD] | [TBD] | 19,721.0 | [TBD] |
| 10 | Warptiling | [TBD] | [TBD] | 21,779.3 | [TBD] |
| 11 | Double Buffering | [TBD] | [TBD] | 17,278.3 | [TBD] |

## 🔍 Key Optimization Techniques Explained

### 1. Naive Implementation
- Each thread computes one output element
- No memory optimization
- Performance: **1.3% of cuBLAS**

**Code**: See [SGEMM_CUDA/src/kernels/1_naive.cuh](SGEMM_CUDA/src/kernels/1_naive.cuh)

### 2. Global Memory Coalescing
- Aligns memory accesses for sequential threads
- Reduces memory transactions from ~548GB to optimal
- **6.4x speedup** over naive

**Key insight**: "Sequential memory accesses by threads in the same warp can be grouped into one transaction"

### 3. Shared Memory Caching
- Uses on-chip shared memory (48KB per SM)
- Blocks data to reduce global memory access
- **1.5x additional speedup**

**Blocking strategy**: Each block loads tiles of A and B into shared memory

### 4. 1D Blocktiling
- Each thread computes multiple output elements (TM=8)
- Increases arithmetic intensity (compute/memory ratio)
- **2.8x additional speedup**

**Formula**: Arithmetic Intensity = 2 * TM * BM * BK / (sizeof(float) * (BM * BK + BK * BN + BM * BN))

### 5. 2D Blocktiling
- Extends to both dimensions (TM=8, TN=8)
- 64 output elements per thread
- **1.9x additional speedup**

### 6. Vectorized Memory Access
- Uses `float4` for 128-bit loads instead of 32-bit
- 4x reduction in load instructions
- **1.1x additional speedup**

**Code pattern**:
```cuda
reinterpret_cast<float4*>(&As[...]) = reinterpret_cast<float4*>(&A[...])[0];
```

### 7-8. Bank Conflict Resolution
- Eliminates shared memory bank conflicts
- Two approaches: linearization and padding
- **Marginal improvement** (already limited by other factors)

### 9. Autotuning
- Optimizes block and tile sizes via grid search
- Best parameters: BM=128, BN=128, BK=8, TM=8, TN=8
- **1.1x additional speedup**

### 10. Warptiling
- Explicit warp-level parallelism
- Reduces register pressure
- Better instruction-level parallelism
- **1.1x additional speedup** → **93.7% of cuBLAS**

**Hierarchy**: Block → Warp → Thread
- Block tiling: Work per SM
- Warp tiling: Work per warp scheduler
- Thread tiling: Work per thread

### 11-12. Double Buffering
- Overlaps compute and memory operations
- Prefetches next tile while computing current
- **Variable results** (sometimes slower due to register pressure)

## 📈 Performance Analysis

### Diminishing Returns

The article demonstrates a classic optimization pattern:
- **First two weekends**: Achieved 80% of cuBLAS performance
- **Next four weekends**: Final 14% improvement
- **Lesson**: Low-level optimization has steep diminishing returns

### H100 vs A6000 Comparison

Expected differences:
- **Peak FP32 Performance**: H100 ~67 TFLOPs vs A6000 ~38 TFLOPs (1.76x)
- **Memory Bandwidth**: H100 ~3.35 TB/s vs A6000 ~768 GB/s (4.36x)
- **Compute Capability**: H100 (9.0) vs A6000 (8.6)

Early kernels (memory-bound) should show larger speedups on H100 due to superior bandwidth.

## 💡 Key Learnings

### From the Article

1. **Memory is the Bottleneck**: "Naive implementation transfers 548GB for operations requiring only 137GB"

2. **Coalescing Matters**: "Sequential memory accesses by threads in the same warp can be grouped"

3. **Arithmetic Intensity**: "Computing multiple outputs per thread dramatically improves compute/memory ratio"

4. **Hardware Hierarchy**: Understanding Block → Warp → Thread mapping is crucial

5. **Autotuning is Essential**: "Optimal parameters vary by GPU architecture"

### From Replication

[To be filled after running on H100]

## 🧪 Verification

### Correctness
All kernels are verified against cuBLAS results with tolerance checks.

### Performance Reproducibility
Results should be within ~5% of reported values (accounting for GPU variance).

## 📚 References

- **Original Article**: [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
- **Source Code**: [github.com/siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
- **NVIDIA cuBLAS**: [docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/)
- **CUDA Programming Guide**: [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 🎯 Assignment Completion Checklist

- [x] Clone and explore the SGEMM_CUDA repository
- [x] Create Modal script for H100 execution
- [ ] Run all kernels on H100 GPU
- [ ] Collect and document performance results
- [ ] Compare H100 results with article's A6000 results
- [ ] Analyze optimization techniques and their impact
- [ ] Document key learnings and insights

---

**Course**: NEU INFO7335 - HPC for AI
**Assignment**: Week 03 - CUDA Matrix Multiplication Optimization
**Reference**: Simon Boehm's CUDA-MMM Worklog
**Date**: February 2026
