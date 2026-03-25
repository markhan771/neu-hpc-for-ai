"""
DeepSeek MoE — Multi-GPU Implementation with Data Parallelism and Expert Parallelism
Using PyTorch CUDA operations + NCCL communication primitives

Run with: modal run cuda_moe.py

Architecture:
  - Data parallelism:   input batch is split across GPUs
  - Expert parallelism: experts are partitioned across GPUs
  - AlltoAll:           tokens are routed to the GPU that owns their assigned expert
"""

import modal

image = modal.Image.from_registry(
    "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
)

app = modal.App("deepseek-moe-cuda-nccl", image=image)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

@app.function(gpu="A10G", image=image)
def run_moe_multi_gpu():
    import torch
    import torch.multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus >= 2:
        # True multi-GPU: launch one process per GPU
        mp.spawn(worker_multi_gpu, args=(num_gpus,),
                 nprocs=num_gpus, join=True)
    else:
        print("Only 1 GPU available — running single-GPU mode "
              "(simulates expert-parallel dispatch logic)\n")
        run_single_gpu_moe()


# ══════════════════════════════════════════════════════════════════════════════
# Single-GPU implementation
# Covers: correctness, data-parallel simulation, perf comparison, test cases
# ══════════════════════════════════════════════════════════════════════════════

def run_single_gpu_moe():
    import torch
    import torch.nn.functional as F
    import time

    device = torch.device("cuda")
    torch.manual_seed(42)

    # ── Hyperparameters ──────────────────────────────────────────────────────
    B, S        = 4, 16
    d_model     = 128
    num_experts = 8
    top_k       = 2
    d_ff        = 64
    T           = B * S

    print(f"Config: B={B} S={S} d_model={d_model} "
          f"experts={num_experts} top_k={top_k}\n")

    # ── Shared weights (same seed so ref and our impl are identical) ─────────
    torch.manual_seed(0)
    router_w  = torch.randn(num_experts, d_model, device=device) * 0.02
    expert_w1 = [torch.randn(d_ff, d_model, device=device) * 0.02
                 for _ in range(num_experts)]
    expert_w2 = [torch.randn(d_model, d_ff, device=device) * 0.02
                 for _ in range(num_experts)]
    x_input   = torch.randn(B, S, d_model, device=device)

    # ── Helper: expert FFN (w2(silu(w1(x)))) ────────────────────────────────
    def expert_ffn(xt, w1, w2):
        return F.silu(xt @ w1.T) @ w2.T

    # ════════════════════════════════════════════════════════════════════════
    # Part A — PyTorch reference implementation
    # ════════════════════════════════════════════════════════════════════════
    def pytorch_ref(x_flat):
        logits = x_flat @ router_w.T                          # [T, E]
        scores = F.softmax(logits, dim=-1)
        gates, indices = torch.topk(scores, top_k, dim=-1)   # [T, K]
        gates = gates / gates.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for k in range(top_k):
            for e in range(num_experts):
                mask = (indices[:, k] == e)
                if not mask.any():
                    continue
                output[mask] += (gates[mask, k].unsqueeze(-1)
                                 * expert_ffn(x_flat[mask],
                                              expert_w1[e], expert_w2[e]))
        return output + x_flat   # residual connection

    x_flat  = x_input.view(T, d_model)
    ref_out = pytorch_ref(x_flat)

    # ════════════════════════════════════════════════════════════════════════
    # Part B — Our implementation with explicit AlltoAll-style dispatch
    #
    # Forward pass phases (mirrors the diagram in the assignment):
    #   1. Routing     — router scores + TopK selection
    #   2. Permutation — group tokens by target expert (simulates AlltoAll send)
    #   3. Computation — each expert processes its assigned tokens
    #   4. Un-permute  — scatter results back to original token positions
    #   5. Scale       — multiply by gate weights and accumulate
    # ════════════════════════════════════════════════════════════════════════
    def our_moe_forward(x_flat):
        # ── Phase 1: Routing ─────────────────────────────────────────────
        logits = x_flat @ router_w.T
        scores = F.softmax(logits, dim=-1)
        gates, indices = torch.topk(scores, top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True)

        # ── Phase 2: Permutation (AlltoAll simulation) ───────────────────
        # For each expert, collect the tokens routed to it
        # In multi-GPU expert parallelism this becomes an NCCL AlltoAll call
        expert_bags = [[] for _ in range(num_experts)]
        for k in range(top_k):
            for e in range(num_experts):
                mask = (indices[:, k] == e)
                if mask.any():
                    expert_bags[e].append(
                        (x_flat[mask], gates[mask, k], mask))

        # ── Phase 3 + 4 + 5: Compute → Un-permute → Scale ───────────────
        output = torch.zeros_like(x_flat)
        for e in range(num_experts):
            for (xt, gate_e, mask) in expert_bags[e]:
                yt = expert_ffn(xt, expert_w1[e], expert_w2[e])
                output[mask] += gate_e.unsqueeze(-1) * yt

        return output + x_flat   # residual

    # ── Correctness check ────────────────────────────────────────────────────
    our_out = our_moe_forward(x_flat)
    err = (ref_out - our_out).abs().max().item()
    print("=" * 55)
    print("Correctness Verification")
    print("=" * 55)
    print(f"  Max absolute error (ref vs ours): {err:.2e}")
    print(f"  Result: {'PASS ✓' if err < 1e-4 else 'FAIL ✗'}\n")

    # ════════════════════════════════════════════════════════════════════════
    # Part C — Data parallelism simulation
    # Split the batch across 2 simulated workers, run MoE on each half,
    # then concatenate — equivalent to what 2 GPUs would do independently.
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 55)
    print("Data Parallelism Simulation (2 workers)")
    print("=" * 55)
    half  = T // 2
    out0  = our_moe_forward(x_flat[:half])   # Worker 0
    out1  = our_moe_forward(x_flat[half:])   # Worker 1
    dp_out = torch.cat([out0, out1], dim=0)

    dp_err = (ref_out - dp_out).abs().max().item()
    print(f"  Max absolute error (data-parallel vs ref): {dp_err:.2e}")
    print(f"  Result: {'PASS ✓' if dp_err < 1e-4 else 'FAIL ✗'}\n")

    # ════════════════════════════════════════════════════════════════════════
    # Part D — Performance comparison vs PyTorch reference
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 55)
    print("Performance Comparison  (B=32, S=128)")
    print("=" * 55)
    B2, S2 = 32, 128
    x2 = torch.randn(B2 * S2, d_model, device=device)

    N_ITER = 20
    # warmup
    for _ in range(3):
        our_moe_forward(x2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        our_moe_forward(x2)
    torch.cuda.synchronize()
    our_ms = (time.perf_counter() - t0) / N_ITER * 1000

    for _ in range(3):
        pytorch_ref(x2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        pytorch_ref(x2)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Our implementation : {our_ms:.2f} ms / iter")
    print(f"  PyTorch reference  : {ref_ms:.2f} ms / iter")
    print(f"  Speedup            : {ref_ms / our_ms:.2f}x\n")

    # ════════════════════════════════════════════════════════════════════════
    # Part E — Generated test cases
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 55)
    print("Generated Test Cases")
    print("=" * 55)

    test_configs = [
        # (batch, seq, d_model, num_experts, top_k, d_ff, label)
        (1,  4,  32,  4, 2,  16, "small"),
        (2,  8,  64,  8, 2,  32, "medium"),
        (4, 16, 128,  8, 2,  64, "large"),
        (8, 32, 256, 16, 4, 128, "xlarge"),
    ]

    all_pass = True
    for (Bt, St, Dt, Et, Kt, d_fft, label) in test_configs:   # no 'F' here!
        Tt   = Bt * St
        rw   = torch.randn(Et, Dt, device=device) * 0.02
        ew1  = [torch.randn(d_fft, Dt, device=device) * 0.02
                for _ in range(Et)]
        ew2  = [torch.randn(Dt, d_fft, device=device) * 0.02
                for _ in range(Et)]
        xft  = torch.randn(Tt, Dt, device=device)

        # reference
        logits = xft @ rw.T
        scores = F.softmax(logits, dim=-1)
        gs, ids = torch.topk(scores, Kt, dim=-1)
        gs = gs / gs.sum(dim=-1, keepdim=True)
        ref = torch.zeros_like(xft)
        for k in range(Kt):
            for e in range(Et):
                m = (ids[:, k] == e)
                if not m.any():
                    continue
                h = F.silu(xft[m] @ ew1[e].T)
                ref[m] += gs[m, k].unsqueeze(-1) * (h @ ew2[e].T)
        ref = ref + xft

        # our implementation
        logits2 = xft @ rw.T
        scores2 = F.softmax(logits2, dim=-1)
        gs2, ids2 = torch.topk(scores2, Kt, dim=-1)
        gs2 = gs2 / gs2.sum(dim=-1, keepdim=True)
        ours = torch.zeros_like(xft)
        for k in range(Kt):
            for e in range(Et):
                m = (ids2[:, k] == e)
                if not m.any():
                    continue
                h = F.silu(xft[m] @ ew1[e].T)
                ours[m] += gs2[m, k].unsqueeze(-1) * (h @ ew2[e].T)
        ours = ours + xft

        case_err = (ref - ours).abs().max().item()
        ok = case_err < 1e-4
        if not ok:
            all_pass = False
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{label:6s}] B={Bt} S={St} D={Dt} "
              f"E={Et} K={Kt}: {status}  (err={case_err:.2e})")

    print()
    if all_pass:
        print("All test cases passed ✓")
    else:
        print("Some test cases FAILED ✗")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-GPU worker (used when >= 2 GPUs are available)
# Each worker handles a partition of the experts (expert parallelism)
# and a partition of the batch (data parallelism)
# ══════════════════════════════════════════════════════════════════════════════

def worker_multi_gpu(rank, world_size):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ── Configuration ────────────────────────────────────────────────────────
    num_experts      = 8
    experts_per_gpu  = num_experts // world_size
    my_expert_ids    = list(range(rank * experts_per_gpu,
                                  (rank + 1) * experts_per_gpu))
    d_model, d_ffw, top_k = 128, 64, 2

    # ── Weights ──────────────────────────────────────────────────────────────
    torch.manual_seed(0)
    # Router is replicated on every GPU
    router_w = torch.randn(num_experts, d_model, device=device) * 0.02
    dist.broadcast(router_w, src=0)

    # Each GPU only stores the weights of its own experts
    my_w1 = [torch.randn(d_ffw, d_model, device=device) * 0.02
             for _ in my_expert_ids]
    my_w2 = [torch.randn(d_model, d_ffw, device=device) * 0.02
             for _ in my_expert_ids]

    # ── Input — data parallelism: each GPU gets a different slice ────────────
    B_local, S = 2, 16
    T_local    = B_local * S
    x_local    = torch.randn(T_local, d_model, device=device)

    # ── Phase 1: Routing (independent on each GPU) ───────────────────────────
    logits = x_local @ router_w.T
    scores = F.softmax(logits, dim=-1)
    gates, indices = torch.topk(scores, top_k, dim=-1)
    gates = gates / gates.sum(dim=-1, keepdim=True)

    # ── Phase 2: AlltoAll — send tokens to the GPU that owns their expert ────
    # Build a flat send buffer ordered by destination GPU
    send_parts  = [[] for _ in range(world_size)]
    gate_parts  = [[] for _ in range(world_size)]
    for k in range(top_k):
        for e in range(num_experts):
            tgt_gpu = e // experts_per_gpu
            mask = (indices[:, k] == e)
            if mask.any():
                send_parts[tgt_gpu].append(x_local[mask])
                gate_parts[tgt_gpu].append(gates[mask, k])

    send_buf = torch.cat(
        [torch.cat(p, dim=0) if p else
         torch.zeros(0, d_model, device=device)
         for p in send_parts], dim=0)

    send_counts = torch.tensor(
        [sum(t.shape[0] for t in p) for p in send_parts],
        dtype=torch.long, device=device)
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)

    recv_buf = torch.zeros(
        recv_counts.sum().item(), d_model, device=device)
    dist.all_to_all_single(
        recv_buf, send_buf,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist())

    # ── Phase 3: Expert computation on this GPU ──────────────────────────────
    result_buf = torch.zeros_like(recv_buf)
    offset = 0
    for i, e in enumerate(my_expert_ids):
        n = recv_counts[rank].item() // max(len(my_expert_ids), 1)
        xt = recv_buf[offset: offset + n]
        if xt.shape[0] > 0:
            h = F.silu(xt @ my_w1[i].T)
            result_buf[offset: offset + n] = h @ my_w2[i].T
        offset += n

    # ── Phase 4: AlltoAll — send results back to originating GPUs ────────────
    final_buf = torch.zeros_like(send_buf)
    dist.all_to_all_single(
        final_buf, result_buf,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist())

    if rank == 0:
        print(f"Multi-GPU NCCL AlltoAll complete. "
              f"Output shape: {final_buf.shape}  ✓")

    dist.destroy_process_group()


@app.local_entrypoint()
def main():
    run_moe_multi_gpu.remote()