# Assignment 1: Matrix Multiplication

This assignment implements single-threaded and multi-threaded matrix multiplication in C using pthreads.

## 📁 Project Structure

```
Assignment1/
├── matrix_mult.h           # Header file with function declarations
├── matrix_mult_single.c    # Single-threaded implementation
├── matrix_mult_multi.c     # Multi-threaded implementation using pthreads
├── test.c                  # Comprehensive test suite
├── benchmark.c             # Performance benchmarking tool
├── Makefile               # Build automation
└── README.md              # This file
```

## 🎯 Assignment Requirements

### ✅ Completed Tasks

1. **Single-threaded matrix multiplication in C**
   - Implemented in `matrix_mult_single.c`
   - Function: `matrix_mult_single(Matrix *A, Matrix *B)`

2. **Comprehensive test cases**
   - Implemented in `test.c`
   - Tests various matrix dimensions including:
     - 1x1 × 1x1 (smallest possible)
     - 1x1 × 1x5 (row vector)
     - 2x1 × 1x3 (rectangular)
     - 2x2 × 2x2 (small square)
     - Various rectangular matrices
     - Large matrices (100×100, 200×150, etc.)
     - Edge cases with large aspect ratios

3. **Multi-threaded version using pthreads**
   - Implemented in `matrix_mult_multi.c`
   - Function: `matrix_mult_multi(Matrix *A, Matrix *B, int num_threads)`
   - Work distribution: rows divided evenly among threads

4. **Verification tests**
   - All test cases verify correctness by comparing single-threaded and multi-threaded results
   - Tests run with 1, 2, 4, and 8 threads

5. **Performance benchmarking**
   - Implemented in `benchmark.c`
   - Measures speedup for thread counts: 1, 4, 16, 32, 64, 128
   - Reports time, speedup, efficiency, and GFLOPS

## 🔧 Building the Project

### Prerequisites
- GCC compiler with pthread support
- Make utility
- Unix-like environment (Linux, macOS, WSL on Windows)

### Compilation

```bash
# Build all executables
make

# Build only tests
make test

# Build only benchmark
make benchmark

# Clean build artifacts
make clean

# Show help
make help
```

## 🧪 Running Tests

```bash
# Run all test cases
make run-test

# Or run directly
./test
```

### Expected Test Output
The test suite will run 18 different test cases covering various matrix dimensions and verify that:
- Single-threaded implementation works correctly
- Multi-threaded implementation produces identical results
- All edge cases are handled properly

Example output:
```
==============================================
Matrix Multiplication Test Suite
==============================================

--- Test 1: 1x1 * 1x1 (smallest) ---
A: 1x1, B: 1x1
...
Test 1: PASSED ✓

--- Test 2: 1x1 * 1x5 (row vector result) ---
...

==============================================
Test Summary
==============================================
Total tests: 18
Passed:      18
Failed:      0
Success rate: 100.0%
==============================================
```

## 📊 Running Performance Benchmarks

```bash
# Run with default matrix size (1000×1000)
make run-benchmark

# Run with custom matrix size
make run-benchmark-2048

# Or run directly
./benchmark
./benchmark 2048
```

### Benchmark Features

The benchmark program provides:

1. **Comprehensive thread scaling analysis**
   - Tests thread counts: 1, 4, 16, 32, 64, 128
   - Reports time, speedup, and efficiency for each

2. **Performance metrics**
   - Execution time
   - Speedup relative to single-threaded
   - Parallel efficiency
   - GFLOPS (billions of floating-point operations per second)

3. **Matrix size scaling test**
   - Tests with sizes: 256, 512, 1024, 2048
   - Shows how performance scales with problem size

### Expected Benchmark Output

```
==============================================
Matrix Multiplication Performance Benchmark
==============================================
Matrix size: 1000x1000
Total operations: 2.00e+09

==============================================
Benchmark: 1000x1000 Matrix Multiplication
==============================================

Threads:   1
  Time:         2.5432 seconds
  Speedup:        1.00x
  GFLOPS:        0.79

Threads:   4
  Time:         0.6891 seconds
  Speedup:        3.69x
  GFLOPS:        2.90

Threads:  16
  Time:         0.2104 seconds
  Speedup:       12.09x
  GFLOPS:        9.51

...

==============================================
Summary Table
==============================================
Threads | Time (s) | Speedup | Efficiency
--------|----------|---------|------------
      1 |   2.5432 |    1.00x |      100.0%
      4 |   0.6891 |    3.69x |       92.2%
     16 |   0.2104 |   12.09x |       75.6%
     32 |   0.1523 |   16.70x |       52.2%
     64 |   0.1402 |   18.14x |       28.3%
    128 |   0.1398 |   18.19x |       14.2%
==============================================
```

## 🧮 Implementation Details

### Matrix Structure
```c
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;
```

### Single-threaded Algorithm
- Standard triple-nested loop (i, j, k)
- Complexity: O(m × n × p) for matrices A(m×n) and B(n×p)

### Multi-threaded Algorithm
- Row-based parallelization
- Each thread computes a range of rows in the result matrix
- Work is divided evenly among threads
- Main thread waits for all worker threads to complete

### Thread Work Distribution
```c
// Example: 10 rows, 3 threads
// Thread 0: rows 0-3 (4 rows)
// Thread 1: rows 4-6 (3 rows)
// Thread 2: rows 7-9 (3 rows)
```

## 📈 Performance Analysis

### Expected Results
- **Linear speedup**: Ideal speedup equals number of threads
- **Reality**: Speedup < number of threads due to:
  - Thread creation/management overhead
  - Cache effects and memory bandwidth
  - Synchronization overhead
  - Diminishing returns with many threads

### Efficiency Factors
- **Small matrices**: Overhead dominates, poor speedup
- **Large matrices**: Better speedup, overhead amortized
- **Thread count**: Sweet spot typically at 4-16 threads for most systems
- **Beyond physical cores**: Hyperthreading provides diminishing returns

## 🔍 Testing Strategy

### Corner Cases Tested
1. **Minimal size**: 1×1 matrices
2. **Vector operations**: 1×n, n×1 matrices
3. **Square matrices**: 2×2, 3×3, 4×4, 100×100
4. **Rectangular matrices**: Various aspect ratios
5. **Large aspect ratios**: 1×100, 100×1, 500×2

### Correctness Verification
- Compare single-threaded vs multi-threaded results
- Floating-point comparison with tolerance (1e-9)
- Test multiple thread counts (1, 2, 4, 8)

## 💡 Usage Examples

### Basic Test
```bash
make run-test
```

### Quick Benchmark
```bash
make run-benchmark
```

### Large Matrix Benchmark
```bash
./benchmark 2048
```

### Custom Testing in Code
```c
Matrix *A = create_matrix(100, 50);
Matrix *B = create_matrix(50, 75);
fill_matrix_random(A);
fill_matrix_random(B);

// Single-threaded
Matrix *C1 = matrix_mult_single(A, B);

// Multi-threaded with 4 threads
Matrix *C2 = matrix_mult_multi(A, B, 4);

// Verify results match
if (compare_matrices(C1, C2, 1e-9)) {
    printf("Results match!\n");
}

free_matrix(A);
free_matrix(B);
free_matrix(C1);
free_matrix(C2);
```

## 📝 Notes

- **Matrix size for benchmarking**: Use at least 1000×1000 for measurable speedup
- **Thread count**: Limited to number of rows (can't use more threads than rows)
- **Memory**: Large matrices may require significant memory (n² × 8 bytes per matrix)
- **Platform**: Performance varies by CPU, core count, and system load

## ✨ Features

- Memory-safe implementation with error checking
- Flexible matrix dimensions
- Comprehensive test coverage
- Detailed performance metrics
- Easy-to-use build system
- Well-documented code

## 🚀 Next Steps

To extend this project, consider:
1. Cache-optimized algorithms (blocking/tiling)
2. SIMD optimizations
3. GPU acceleration with CUDA
4. Strassen's algorithm for large matrices
5. Sparse matrix support

---

**Author**: Assignment 1 Implementation
**Course**: INFO7335
**Date**: February 2026
