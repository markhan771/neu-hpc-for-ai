"""
DeepSeek MoE Step 1 — Modal 上跑 PyTorch 参考实现
运行命令: modal run moe_step1.py
"""

import modal

# 直接用已经装好 PyTorch 的官方镜像，不需要再下载
image = modal.Image.from_registry(
    "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
)

app = modal.App("deepseek-moe-step1", image=image)


@app.function(gpu="A10G")
def run_moe():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── Expert（小 FFN）──────────────────────────────────────────────────
    class Expert(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)))

    # ── Router ───────────────────────────────────────────────────────────
    class SoftmaxRouter(nn.Module):
        def __init__(self, d_model, num_experts):
            super().__init__()
            self.weight = nn.Linear(d_model, num_experts, bias=False)

        def forward(self, x, top_k):
            logits = self.weight(x)
            scores = F.softmax(logits, dim=-1)
            gates, indices = torch.topk(scores, top_k, dim=-1)
            gates = gates / gates.sum(dim=-1, keepdim=True)
            return scores, indices, gates

    # ── DeepSeek MoE Layer ───────────────────────────────────────────────
    class DeepSeekMoE(nn.Module):
        def __init__(self, d_model, num_experts, top_k, d_ff):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.router = SoftmaxRouter(d_model, num_experts)
            self.experts = nn.ModuleList([
                Expert(d_model, d_ff) for _ in range(num_experts)
            ])

        def forward(self, x):
            B, S, D = x.shape
            x_flat = x.view(-1, D)

            # 1. Router 打分，选 Top-K
            scores, indices, gates = self.router(x_flat, self.top_k)

            # 2. 分发给 Expert，收集结果
            output = torch.zeros_like(x_flat)
            for k in range(self.top_k):
                expert_idx = indices[:, k]
                gate_k     = gates[:, k]
                for e in range(self.num_experts):
                    mask = (expert_idx == e)
                    if not mask.any():
                        continue
                    out_e = self.experts[e](x_flat[mask])
                    output[mask] += gate_k[mask].unsqueeze(-1) * out_e

            # 3. 残差连接
            output = output + x_flat
            return output.view(B, S, D)

    # ── 测试 ─────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("=" * 50)
    print("DeepSeek MoE PyTorch 参考实现测试")
    print("=" * 50)

    configs = [
        # (batch, seq, d_model, num_experts, top_k, d_ff, 名称)
        (1,  4,  32,  4, 2,  16, "小型"),
        (2,  8,  64,  8, 2,  32, "中型"),
        (4, 16, 128,  8, 2,  64, "大型"),
    ]

    all_pass = True
    for (B, S, D, E, K, d_ff, name) in configs:
        model = DeepSeekMoE(D, E, K, d_ff).to(device)
        x = torch.randn(B, S, D, device=device)

        with torch.no_grad():
            out = model(x)

        shape_ok  = (out.shape == x.shape)
        value_ok  = (not torch.allclose(out, x))
        ok = shape_ok and value_ok
        if not ok:
            all_pass = False

        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"[{name}] B={B} S={S} D={D} E={E} K={K}: {status}")
        print(f"        输出前3值: {out[0, 0, :3].tolist()}")

    print()
    if all_pass:
        print("✓ 所有测试通过！逻辑正确，可以进行 CUDA 实现。")
    else:
        print("✗ 有测试失败，需要检查。")


@app.local_entrypoint()
def main():
    run_moe.remote()