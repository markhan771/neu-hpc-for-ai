THIS IS REPO FOR INFO7335

# CUDA GEMM Kernel (INFO7335)

## Overview
This project implements a naive CUDA GEMM (Generalized Matrix Multiplication) kernel using only global memory. 
No high-level CUDA libraries (e.g., cuBLAS, cuDNN) or advanced optimizations (tiling, shared memory) are used.

The kernel computes:
C ← α · op(A) · op(B) + β · C

where op(A) and op(B) can optionally be transposed.

---

## Implementation Details
- Each CUDA thread computes one output element C[i, j].
- Grid and block structure: 2D grid with 16×16 thread blocks.
- Supports optional transposition of A and/or B.
- Updates matrix C in place.
- A CPU reference implementation is used to verify correctness.

---

## Files
- `gemm.cu`  
  CUDA kernel implementation and CPU reference tests.
- `modal_gemm.py`  
  Script for compiling and running the CUDA code on a cloud GPU using Modal.

---

## How to Run (Modal)
This project was tested on a GPU using Modal.

```bash
modal run modal_gemm.py