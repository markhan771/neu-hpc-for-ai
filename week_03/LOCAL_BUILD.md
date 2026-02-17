# Local Build Instructions

If you have a local NVIDIA GPU, you can build and run the kernels locally.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+ (12.x recommended)
- CMake 3.19+
- C++ compiler with C++20 support
- Git

## Build Steps

```bash
cd week_03/SGEMM_CUDA

# 1. Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# 2. Update CMakeLists.txt with your GPU's compute capability
# For example:
# - RTX 3080/3090: 86
# - RTX 4080/4090: 89
# - A100: 80
# - H100: 90
# Edit CMakeLists.txt line 13:
# set(CUDA_COMPUTE_CAPABILITY XX)

# 3. Build
mkdir build
cd build
cmake ..
cmake --build . -j

# 4. Run all kernels (benchmarks at 4096x4096)
./sgemm

# 5. Run a specific kernel
./sgemm 0   # cuBLAS baseline
./sgemm 1   # Naive
./sgemm 2   # Global memory coalescing
# ... etc
./sgemm 10  # Warptiling (best optimized)
```

## Expected Output

```
-------------------------------------------------------------
M=N=K=4096, alpha=1.0, beta=0.0
Kernel: [Kernel Name]
Time: X.XXX ms
Performance: XXXX.X GFLOPs/s
Performance relative to cuBLAS: XX.X%
-------------------------------------------------------------
```

## Profiling with NVIDIA Nsight Compute

```bash
# Profile a specific kernel
ncu --set full -o profile_k10 ./sgemm 10

# View the profile
ncu-ui profile_k10.ncu-rep
```

## Troubleshooting

### Build Errors

**Error**: `nvcc: command not found`
- **Fix**: Install CUDA Toolkit and add to PATH
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

**Error**: `compute_XX is not supported`
- **Fix**: Update `CUDA_COMPUTE_CAPABILITY` in `CMakeLists.txt` to match your GPU

**Error**: `undefined reference to cublas...`
- **Fix**: Ensure CUDA Toolkit includes cuBLAS library

### Runtime Errors

**Error**: `CUDA error: out of memory`
- **Fix**: The default matrix size is 4096x4096. For GPUs with less memory, edit `sgemm.cu` to use smaller sizes (e.g., 2048)

**Error**: `Invalid device function`
- **Fix**: Rebuild with correct compute capability for your GPU

## Performance Tips

1. **Close other applications** to ensure GPU is not busy
2. **Run multiple times** and take the average
3. **Warm up** the GPU by running once before timing
4. **Use power mode** on laptops (not battery saving)

## Comparing Results

After running locally, you can compare your results with:
- **Article's A6000 results**: See main README.md
- **Your H100 results**: Run `modal run run_on_h100.py`

Different GPUs will show different performance characteristics:
- **Memory-bound kernels** (1-3): Performance scales with memory bandwidth
- **Compute-bound kernels** (4-10): Performance scales with FP32 throughput

## Next Steps

1. Run all kernels locally
2. Record your GPU's results
3. Run on H100 via Modal
4. Compare and analyze the differences
5. Update README.md with findings
