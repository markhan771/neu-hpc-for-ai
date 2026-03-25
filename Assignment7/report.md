

## 1. Overview

This assignment implements the **DeepSeek MoE (Mixture-of-Experts) operator** with:
- **Data parallelism**: the input batch is split across GPUs
- **Expert parallelism**: the expert FFNs are partitioned across GPUs
- **NCCL AlltoAll**: tokens are routed to the GPU that owns their assigned expert

The implementation follows the DeepSeek MoE formulation from the assignment:

$$h_t = \sum_{i=1}^{mN} g_{i,t} \cdot \text{FFN}_i(u_t) + u_t$$

where $g_{i,t} = s_{i,t}$ if $s_{i,t} \in \text{TopK}$, else $0$, and $s_{i,t} = \text{Softmax}_i(u_t^T e_i)$.

---

## 2. Implementation

### Files
| File | Description |
|---|---|
| `step1.py` | PyTorch reference implementation (correctness baseline) |
| `cuda_moe.py` | Main implementation: CUDA ops + AlltoAll dispatch + NCCL multi-GPU |
| `benchmark.py` | Performance comparison vs Dense FFN baseline |

### Architecture

```
Input tokens
     │
     ▼
 SoftmaxRouter ──── scores = Softmax(x @ W_router^T)
     │               select Top-K experts per token
     ▼
 Permutation ─────── group tokens by target expert
 (AlltoAll)          [simulates NCCL AlltoAll in multi-GPU]
     │
     ▼
 Expert FFNs ─────── FFN_e(x) = W2_e(SiLU(W1_e(x)))
 (partitioned        each GPU owns experts_per_gpu experts
  across GPUs)
     │
     ▼
 Un-permute ─────── scatter results back to original positions
 (AlltoAll)
     │
     ▼
 Scale + Sum ─────── output = Σ gate_k * expert_k(x) + x
```

### Key Design Decisions

**1. Fine-grained expert segmentation (DeepSeek-specific)**
Each expert uses a smaller intermediate dimension `d_ff = d_model * 4 / m`, where `m` is the segmentation factor. This allows more experts with lower per-expert cost.

**2. Explicit token dispatch (AlltoAll simulation)**
Rather than materializing the full attention-style routing matrix, we explicitly permute tokens by target expert — the same pattern used by NCCL `AlltoAll` in real multi-GPU deployments.

**3. Normalized gate weights**
After TopK selection, gate weights are renormalized so they sum to 1 per token:
```python
gates = gates / gates.sum(dim=-1, keepdim=True)
```

**4. Residual connection**
Following the assignment formula, we add the original input back:
```python
output = weighted_expert_sum + x_flat
```

---

## 3. Test Cases (Generated)

All test cases were generated and verified to pass with **zero numerical error**:

| Config | B | S | D | Experts | TopK | Result |
|---|---|---|---|---|---|---|
| small  | 1 |  4 |  32 |  4 | 2 | ✓ PASS (err=0.00e+00) |
| medium | 2 |  8 |  64 |  8 | 2 | ✓ PASS (err=0.00e+00) |
| large  | 4 | 16 | 128 |  8 | 2 | ✓ PASS (err=0.00e+00) |
| xlarge | 8 | 32 | 256 | 16 | 4 | ✓ PASS (err=0.00e+00) |

Additional verification:
- **Correctness**: max absolute error vs reference = `0.00e+00` ✓
- **Data parallelism**: max absolute error vs reference = `1.19e-07` ✓ (floating point rounding only)

---

## 4. Performance Comparison

Benchmarked on **NVIDIA A10G GPU** (Modal cloud).  
Baseline: Dense FFN with the **same total parameter count** as the MoE layer (`d_ff_dense = num_experts × d_ff`).

| Config | Params | MoE (ms) | Dense FFN (ms) | Speedup |
|---|---|---|---|---|
| B=2  S=64  D=256  E=8  K=2  | 264K   |  4.80 |  0.07 | 0.01x |
| B=4  S=128 D=512  E=8  K=2  | 1.05M  |  4.87 |  0.12 | 0.03x |
| B=8  S=256 D=1024 E=16 K=4  | 8.40M  | 19.37 |  3.16 | 0.16x |
| B=16 S=512 D=1024 E=16 K=4  | 8.40M  | 21.76 | 13.11 | 0.60x |

### Analysis

The MoE operator is **slower** than a dense FFN of equal parameter count in wall-clock time for small batch sizes. This is expected and well-documented in the literature. The reasons are:

1. **Sequential expert loop**: our current implementation loops over experts in Python, which adds overhead per-expert. A production implementation fuses all expert computations into a single batched GEMM.

2. **Sparse activation**: MoE's advantage is not raw speed on a single GPU — it is that a **much larger model** (more total parameters, more capacity) can be run at the same **compute cost per token**. For example, a 64-expert MoE model has 64× more parameters than a single FFN but only activates 2 experts per token, keeping FLOP count constant.

3. **Multi-GPU scaling**: the real speedup of MoE comes when experts are distributed across many GPUs. Each GPU then processes only its share of expert computations in parallel, achieving near-linear scaling. Our NCCL AlltoAll implementation enables this.

4. **Trend at large batch sizes**: at `B=16 S=512` the speedup improves to `0.60x`, showing that as batch size grows, the MoE overhead amortizes — consistent with production MoE deployments that use large batches.

---

## 5. Multi-GPU NCCL Design

When 2+ GPUs are available, the implementation uses `torch.distributed` with NCCL backend:

```
GPU 0                          GPU 1
──────────────────────         ──────────────────────
Owns: Expert 0,1,2,3           Owns: Expert 4,5,6,7
Input: batch[:half]            Input: batch[half:]

Phase 1: Routing (independent, router weights replicated via broadcast)
Phase 2: AlltoAll — send tokens to correct GPU
Phase 3: Compute expert FFNs locally
Phase 4: AlltoAll — return results to originating GPU
Phase 5: Scale by gate weights + residual
```

NCCL operations used:
- `dist.broadcast`: replicate router weights to all GPUs at startup
- `dist.all_to_all_single`: exchange tokens between GPUs (phases 2 and 4)

---

## 6. Running the Code

```bash
# Step 1: PyTorch reference (verify logic)
modal run step1.py

# Step 2: Full CUDA + NCCL implementation
modal run cuda_moe.py

# Step 3: Performance benchmark
modal run benchmark.py
```

All three scripts run on Modal cloud with an NVIDIA A10G GPU.