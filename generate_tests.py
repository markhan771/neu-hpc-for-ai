import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
#  配置（小型，方便测试；和真实 V3 结构完全一致）
# ══════════════════════════════════════════════════════════
class MiniConfig:
    hidden_size           = 16   # H：token 向量维度
    moe_intermediate_size =  8   # I：每个专家 FFN 中间层维度
    n_routed_experts      =  8   # E：路由专家总数
    num_experts_per_tok   =  2   # K：每个 token 激活的专家数
    n_shared_experts      =  1   # Ns：共享专家数
    n_group               =  2   # 路由分组数
    topk_group            =  1   # 选择的组数
    routed_scaling_factor = 1.0  # 路由权重缩放系数
    norm_topk_prob        = True # 是否对 topk 权重归一化
    scoring_func          = "sigmoid"
    topk_method           = "noaux_tc"
    hidden_act            = "silu"


# ══════════════════════════════════════════════════════════
#  单个 FFN 专家
#  forward: down_proj( silu(gate_proj(x)) * up_proj(x) )
# ══════════════════════════════════════════════════════════
class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ══════════════════════════════════════════════════════════
#  MoE 路由器（Gate）
#  实现 noaux_tc 分组 top-K 路由
# ══════════════════════════════════════════════════════════
class MoEGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K                       = cfg.num_experts_per_tok
        self.E                       = cfg.n_routed_experts
        self.routed_scaling_factor   = cfg.routed_scaling_factor
        self.norm_topk_prob          = cfg.norm_topk_prob
        self.n_group                 = cfg.n_group
        self.topk_group              = cfg.topk_group
        # 路由权重矩阵：每行是一个专家的"质心向量"
        self.weight = nn.Parameter(
            torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        # 负载均衡偏置（noaux_tc 专用）
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        N = B * S
        x = hidden_states.view(N, H).float()

        # ① 线性投影 + sigmoid 得到亲和力分数
        logits = F.linear(x, self.weight.float())   # (N, E)
        scores = torch.sigmoid(logits)              # (N, E)

        # ② 加偏置（只用于路由决策，不用于最终权重）
        sfc = scores + self.e_score_correction_bias.unsqueeze(0)  # (N, E)

        # ③ 分组：每组取 top-2 求和 → group_scores
        E_per_g = self.E // self.n_group
        group_scores = (
            sfc.view(N, self.n_group, E_per_g)
               .topk(2, dim=-1)[0]
               .sum(dim=-1)
        )  # (N, n_group)

        # ④ 选出得分最高的 topk_group 个组
        group_idx  = torch.topk(group_scores, k=self.topk_group,
                                dim=-1, sorted=False)[1]  # (N, topk_group)
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)

        # ⑤ 展开 mask 到每个专家，遮住不在选中组里的专家
        score_mask = (
            group_mask.unsqueeze(-1)
                      .expand(N, self.n_group, E_per_g)
                      .reshape(N, self.E)
        )
        tmp = sfc.masked_fill(~score_mask.bool(), float("-inf"))

        # ⑥ 在候选专家中取全局 top-K
        _, topk_idx = torch.topk(tmp, k=self.K, dim=-1, sorted=False)  # (N, K)

        # ⑦ 用原始 scores（不含偏置）作为权重
        topk_weight = scores.gather(1, topk_idx)  # (N, K)

        # ⑧ 归一化 + 缩放
        if self.K > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight  # 均为 (N, K)


# ══════════════════════════════════════════════════════════
#  完整 MoE 层
#  output = input + Σ shared_expert(input)
#                 + Σ_k g_k * routed_expert_k(input)
# ══════════════════════════════════════════════════════════
class DeepseekV3MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.num_experts_per_tok

        # 路由专家
        self.experts = nn.ModuleList([
            DeepseekV3MLP(cfg.hidden_size, cfg.moe_intermediate_size)
            for _ in range(cfg.n_routed_experts)
        ])

        # 路由器
        self.gate = MoEGate(cfg)

        # 共享专家（中间层更宽：I * Ns）
        shared_interm = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_experts = DeepseekV3MLP(cfg.hidden_size, shared_interm)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        N = B * S
        identity    = hidden_states
        hidden_flat = hidden_states.view(N, H)

        # ── 路由决策 ──
        topk_idx, topk_weight = self.gate(hidden_states)  # (N,K), (N,K)

        # ── 路由专家加权求和 ──
        y = torch.zeros(N, H)
        for k in range(self.K):
            eids    = topk_idx[:, k]      # (N,) 每个 token 选的第 k 个专家编号
            weights = topk_weight[:, k]   # (N,) 对应权重
            for eid in range(len(self.experts)):
                mask = (eids == eid)
                if mask.any():
                    out = self.experts[eid](hidden_flat[mask])
                    y[mask] += weights[mask].unsqueeze(-1) * out

        # ── 共享专家（全部 token 都过） ──
        y += self.shared_experts(hidden_flat)

        # ── 残差连接 ──
        return (identity + y.view(B, S, H)).to(hidden_states.dtype)


# ══════════════════════════════════════════════════════════
#  生成测试用例并保存
# ══════════════════════════════════════════════════════════
def t2l(tensor):
    """Tensor → Python list（float32）"""
    return tensor.detach().float().numpy().tolist()


def generate(n_cases=6, seed=42):
    torch.manual_seed(seed)
    cfg   = MiniConfig()
    model = DeepseekV3MoE(cfg)
    model.eval()

    cases = []
    for i in range(n_cases):
        S = torch.randint(1, 5, (1,)).item()
        x = torch.randn(1, S, cfg.hidden_size)

        with torch.no_grad():
            out = model(x)

            # 记录路由器中间值（用于子模块验证）
            x_flat  = x.view(-1, cfg.hidden_size).float()
            logits  = F.linear(x_flat, model.gate.weight.float())
            scores  = torch.sigmoid(logits)
            tidx, tw = model.gate(x)

        cases.append({
            "id":           i,
            "B":            1,
            "S":            int(S),
            "H":            cfg.hidden_size,
            "input":        t2l(x),
            "output":       t2l(out),
            "gate_logits":  t2l(logits),
            "gate_scores":  t2l(scores),
            "topk_idx":     t2l(tidx.int()),
            "topk_weight":  t2l(tw),
        })
        print(f"  Case {i}: input(1,{S},{cfg.hidden_size}) "
              f"→ output(1,{S},{cfg.hidden_size})")

    # ── 权重 ──
    weights = {}
    # 路由器
    weights["gate.weight"] = t2l(model.gate.weight)
    weights["gate.e_score_correction_bias"] = t2l(
        model.gate.e_score_correction_bias)
    # 路由专家
    for i, exp in enumerate(model.experts):
        weights[f"experts.{i}.gate_proj.weight"] = t2l(exp.gate_proj.weight)
        weights[f"experts.{i}.up_proj.weight"]   = t2l(exp.up_proj.weight)
        weights[f"experts.{i}.down_proj.weight"] = t2l(exp.down_proj.weight)
    # 共享专家
    weights["shared_experts.gate_proj.weight"] = t2l(
        model.shared_experts.gate_proj.weight)
    weights["shared_experts.up_proj.weight"]   = t2l(
        model.shared_experts.up_proj.weight)
    weights["shared_experts.down_proj.weight"] = t2l(
        model.shared_experts.down_proj.weight)

    result = {
        "config": {
            "hidden_size":           cfg.hidden_size,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "n_routed_experts":      cfg.n_routed_experts,
            "num_experts_per_tok":   cfg.num_experts_per_tok,
            "n_shared_experts":      cfg.n_shared_experts,
            "n_group":               cfg.n_group,
            "topk_group":            cfg.topk_group,
            "routed_scaling_factor": cfg.routed_scaling_factor,
            "norm_topk_prob":        int(cfg.norm_topk_prob),
        },
        "weights":     weights,
        "test_cases":  cases,
    }

    with open("test_cases.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 已生成 {n_cases} 个测试用例 → test_cases.json")


if __name__ == "__main__":
    print("生成 DeepSeek-V3 MoE 测试用例...\n")
    generate(n_cases=6)