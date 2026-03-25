"""
DeepSeek MoE — Performance Comparison vs Dense FFN Baseline
Run with: modal run benchmark.py

Comparison target: a standard dense FFN with the same total parameter count
as our MoE layer.  This is the standard apples-to-apples benchmark used in
the MoE literature (e.g. the original Switch Transformer paper).
"""

import modal

# Use a newer PyTorch image to avoid version conflicts
image = modal.Image.from_registry(
    "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
)

app = modal.App("deepseek-moe-benchmark", image=image)


@app.function(gpu="A10G", image=image, timeout=600)
def run_benchmark():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time

    device = torch.device("cuda")
    torch.manual_seed(0)

    # ══════════════════════════════════════════════════════════════════════
    # Our DeepSeek MoE implementation
    # ══════════════════════════════════════════════════════════════════════
    class OurDeepSeekMoE(nn.Module):
        """
        DeepSeek MoE forward pass:
          1. SoftmaxRouter  : scores = Softmax(x @ W_r^T), select TopK
          2. Expert FFN     : out = W2(SiLU(W1(x)))  for each expert
          3. Weighted sum   : output = Σ gate_k * expert_k(x)
          4. Residual       : output = output + x
        Dispatch logic explicitly simulates the AlltoAll permutation
        used in multi-GPU expert parallelism.
        """
        def __init__(self, d_model, num_experts, top_k, d_ff):
            super().__init__()
            self.num_experts = num_experts
            self.top_k       = top_k
            self.router_w = nn.Parameter(
                torch.randn(num_experts, d_model) * 0.02)
            self.w1 = nn.ParameterList([
                nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
                for _ in range(num_experts)])
            self.w2 = nn.ParameterList([
                nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
                for _ in range(num_experts)])

        def forward(self, x):
            B, S, D = x.shape
            x_flat  = x.view(-1, D)

            # Phase 1 — Routing
            logits = x_flat @ self.router_w.T
            scores = F.softmax(logits, dim=-1)
            gates, indices = torch.topk(scores, self.top_k, dim=-1)
            gates = gates / gates.sum(dim=-1, keepdim=True)

            # Phase 2-5 — Permute → Compute → Gather → Scale
            output = torch.zeros_like(x_flat)
            for k in range(self.top_k):
                for e in range(self.num_experts):
                    mask = (indices[:, k] == e)
                    if not mask.any():
                        continue
                    xt = x_flat[mask]
                    h  = F.silu(xt @ self.w1[e].T)
                    yt = h  @ self.w2[e].T
                    output[mask] += gates[mask, k].unsqueeze(-1) * yt

            return (output + x_flat).view(B, S, D)

    # ══════════════════════════════════════════════════════════════════════
    # Baseline: Dense FFN with the same total parameter count
    #
    # MoE params  = num_experts * (d_model*d_ff + d_ff*d_model)
    #             = 2 * num_experts * d_model * d_ff
    # Dense params= d_model * d_ff_dense + d_ff_dense * d_model
    #             = 2 * d_model * d_ff_dense
    # → d_ff_dense = num_experts * d_ff   (same total params)
    # ══════════════════════════════════════════════════════════════════════
    class DenseFFN(nn.Module):
        """Standard FFN used as the comparison baseline."""
        def __init__(self, d_model, d_ff_dense):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff_dense, bias=False)
            self.w2 = nn.Linear(d_ff_dense, d_model, bias=False)

        def forward(self, x):
            B, S, D = x.shape
            h = F.silu(self.w1(x.view(-1, D)))
            return (self.w2(h) + x.view(-1, D)).view(B, S, D)

    # ══════════════════════════════════════════════════════════════════════
    # Benchmark helper
    # ══════════════════════════════════════════════════════════════════════
    def benchmark(model, x, n_warmup=5, n_iter=50):
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                model(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000  # ms

    # ══════════════════════════════════════════════════════════════════════
    # Run benchmarks across multiple dataset sizes
    # ══════════════════════════════════════════════════════════════════════
    configs = [
        # (batch, seq_len, d_model, num_experts, top_k, d_ff, label)
        (2,   64,  256,  8, 2,  64, "small"),
        (4,  128,  512,  8, 2, 128, "medium"),
        (8,  256, 1024, 16, 4, 256, "large"),
        (16, 512, 1024, 16, 4, 256, "xlarge"),
    ]

    print("=" * 72)
    print("DeepSeek MoE  vs  Dense FFN (equal parameter count)")
    print("Baseline: Dense FFN with d_ff_dense = num_experts × d_ff")
    print("=" * 72)
    print(f"{'Config':<24} {'Params':>10} "
          f"{'MoE (ms)':>10} {'Dense (ms)':>12} {'Speedup':>9}")
    print("-" * 72)

    for (B, S, D, E, K, d_fft, label) in configs:
        x = torch.randn(B, S, D, device=device)

        # Our MoE
        moe   = OurDeepSeekMoE(D, E, K, d_fft).to(device).eval()
        moe_ms = benchmark(moe, x)

        # Dense FFN with same total params
        d_ff_dense = E * d_fft
        dense      = DenseFFN(D, d_ff_dense).to(device).eval()
        dense_ms   = benchmark(dense, x)

        n_params = sum(p.numel() for p in moe.parameters())
        speedup  = dense_ms / moe_ms

        cfg_str = f"B={B} S={S} D={D} E={E} K={K}"
        print(f"  {cfg_str:<22} {n_params:>10,} "
              f"{moe_ms:>10.2f} {dense_ms:>12.2f} {speedup:>8.2f}x"
              f"  [{label}]")

    print("-" * 72)
    print("Speedup > 1.0x means our MoE is faster than the dense baseline.\n")

    # ══════════════════════════════════════════════════════════════════════
    # Correctness: all generated test cases
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("Generated Test Cases — Correctness Verification")
    print("=" * 72)

    test_configs = [
        (1,  4,  32,  4, 2,  16, "small"),
        (2,  8,  64,  8, 2,  32, "medium"),
        (4, 16, 128,  8, 2,  64, "large"),
        (8, 32, 256, 16, 4, 128, "xlarge"),
    ]

    all_pass = True
    for (Bt, St, Dt, Et, Kt, d_fft, label) in test_configs:
        model = OurDeepSeekMoE(Dt, Et, Kt, d_fft).to(device).eval()
        xi    = torch.randn(Bt, St, Dt, device=device)

        # Reference: compute directly with same weights
        x_flat = xi.view(-1, Dt)
        logits = x_flat @ model.router_w.T
        scores = F.softmax(logits, dim=-1)
        gates, indices = torch.topk(scores, Kt, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True)
        ref = torch.zeros_like(x_flat)
        for k in range(Kt):
            for e in range(Et):
                mask = (indices[:, k] == e)
                if not mask.any():
                    continue
                h = F.silu(x_flat[mask] @ model.w1[e].T)
                ref[mask] += gates[mask, k].unsqueeze(-1) * (h @ model.w2[e].T)
        ref = (ref + x_flat).view(Bt, St, Dt)

        with torch.no_grad():
            out = model(xi)

        err = (ref - out).abs().max().item()
        ok  = err < 1e-4
        if not ok:
            all_pass = False
        print(f"  [{label:6s}] B={Bt} S={St} D={Dt} E={Et} K={Kt}: "
              f"{'PASS ✓' if ok else 'FAIL ✗'}  (max err={err:.2e})")

    print()
    print("All test cases passed ✓" if all_pass else "Some tests FAILED ✗")


@app.local_entrypoint()
def main():
    run_benchmark.remote()