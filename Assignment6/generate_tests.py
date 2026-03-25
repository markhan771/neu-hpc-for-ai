"""
DeepSeek-V3 MoE Test Case Generator
Week 7 Assignment

Implements a minimal DeepSeek-V3 MoE layer in PyTorch and generates
ground-truth test cases (inputs + expected outputs) for validating
the pure-C implementation.

Usage:
    pip install torch
    python generate_test_cases.py
    → writes test_cases.json
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model configuration (miniaturised for fast testing) ───────────────────────
class MiniConfig:
    hidden_size           = 16   # H: token embedding dimension
    moe_intermediate_size =  8   # I: FFN intermediate dimension per expert
    n_routed_experts      =  8   # E: total number of routed experts
    num_experts_per_tok   =  2   # K: experts activated per token
    n_shared_experts      =  1   # Ns: shared experts (applied to every token)
    n_group               =  2   # number of expert groups for grouped routing
    topk_group            =  1   # number of groups selected per token
    routed_scaling_factor = 1.0  # scale factor applied to routed expert weights
    norm_topk_prob        = True # whether to normalise top-K weights to sum to 1
    scoring_func          = "sigmoid"
    topk_method           = "noaux_tc"
    hidden_act            = "silu"


# ── Single expert FFN ──────────────────────────────────────────────────────────
# Uses the SwiGLU activation variant adopted by DeepSeek-V3:
#   output = down_proj( silu(gate_proj(x)) * up_proj(x) )
class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: element-wise gate controlled by silu(gate_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── MoE router (gate) ─────────────────────────────────────────────────────────
# Implements the noaux_tc grouped top-K routing used in DeepSeek-V3.
#
# Key design decisions vs a vanilla softmax router:
#   1. sigmoid (not softmax) scores — each expert scored independently
#   2. a learnable correction bias added only for routing decisions,
#      NOT included in the final gate weights
#   3. grouped routing: experts are divided into n_group groups;
#      only topk_group groups are eligible, limiting cross-group traffic
class MoEGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K                       = cfg.num_experts_per_tok
        self.E                       = cfg.n_routed_experts
        self.routed_scaling_factor   = cfg.routed_scaling_factor
        self.norm_topk_prob          = cfg.norm_topk_prob
        self.n_group                 = cfg.n_group
        self.topk_group              = cfg.topk_group
        # Weight matrix: each row is the centroid vector of one expert
        self.weight = nn.Parameter(
            torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        # Load-balancing correction bias (noaux_tc specific)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        N = B * S                              # total tokens
        x = hidden_states.view(N, H).float()

        # Step 1: linear projection + sigmoid → per-expert affinity scores
        logits = F.linear(x, self.weight.float())   # (N, E)
        scores = torch.sigmoid(logits)              # (N, E)

        # Step 2: add correction bias (used only for routing, not for weights)
        sfc = scores + self.e_score_correction_bias.unsqueeze(0)  # (N, E)

        # Step 3: grouped scoring — within each group take top-2 and sum
        E_per_g = self.E // self.n_group
        group_scores = (
            sfc.view(N, self.n_group, E_per_g)
               .topk(2, dim=-1)[0]
               .sum(dim=-1)
        )  # (N, n_group)

        # Step 4: select the topk_group highest-scoring groups
        group_idx  = torch.topk(group_scores, k=self.topk_group,
                                dim=-1, sorted=False)[1]  # (N, topk_group)
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)

        # Step 5: expand group mask to individual experts; mask out non-selected groups
        score_mask = (
            group_mask.unsqueeze(-1)
                      .expand(N, self.n_group, E_per_g)
                      .reshape(N, self.E)
        )
        tmp = sfc.masked_fill(~score_mask.bool(), float("-inf"))

        # Step 6: global top-K among the candidate experts
        _, topk_idx = torch.topk(tmp, k=self.K, dim=-1, sorted=False)  # (N, K)

        # Step 7: gate weights come from the ORIGINAL scores (no bias)
        topk_weight = scores.gather(1, topk_idx)  # (N, K)

        # Step 8: normalise so weights sum to 1 per token, then scale
        if self.K > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight   # both (N, K)


# ── Full MoE layer ────────────────────────────────────────────────────────────
# Forward pass:
#   output = input                                  (residual connection)
#          + sum_k  gate_k * routed_expert_k(input) (top-K routed experts)
#          + shared_expert(input)                   (applied to every token)
class DeepseekV3MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg.num_experts_per_tok

        # Routed experts
        self.experts = nn.ModuleList([
            DeepseekV3MLP(cfg.hidden_size, cfg.moe_intermediate_size)
            for _ in range(cfg.n_routed_experts)
        ])

        # Router / gate
        self.gate = MoEGate(cfg)

        # Shared expert — wider intermediate layer (I * Ns)
        shared_interm = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_experts = DeepseekV3MLP(cfg.hidden_size, shared_interm)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        N = B * S
        identity    = hidden_states
        hidden_flat = hidden_states.view(N, H)

        # Routing decision
        topk_idx, topk_weight = self.gate(hidden_states)  # (N,K), (N,K)

        # Weighted sum over routed experts
        y = torch.zeros(N, H)
        for k in range(self.K):
            eids    = topk_idx[:, k]     # expert index for k-th choice, shape (N,)
            weights = topk_weight[:, k]  # corresponding gate weight,   shape (N,)
            for eid in range(len(self.experts)):
                mask = (eids == eid)
                if mask.any():
                    out = self.experts[eid](hidden_flat[mask])
                    y[mask] += weights[mask].unsqueeze(-1) * out

        # Shared expert applied to all tokens
        y += self.shared_experts(hidden_flat)

        # Residual connection
        return (identity + y.view(B, S, H)).to(hidden_states.dtype)


# ── Helpers ───────────────────────────────────────────────────────────────────
def t2l(tensor):
    """Convert a tensor to a plain Python list of float32 values."""
    return tensor.detach().float().numpy().tolist()


# ── Test case generation ──────────────────────────────────────────────────────
def generate(n_cases=6, seed=42):
    """
    Build a MoE model with fixed random weights, run n_cases forward passes
    with random inputs, and save everything (weights + inputs + outputs) to
    test_cases.json.

    The JSON file is then consumed by the pure-C implementation to verify
    that it produces numerically matching outputs.
    """
    torch.manual_seed(seed)
    cfg   = MiniConfig()
    model = DeepseekV3MoE(cfg)
    model.eval()

    cases = []
    for i in range(n_cases):
        # Random sequence length between 1 and 4
        S = torch.randint(1, 5, (1,)).item()
        x = torch.randn(1, S, cfg.hidden_size)

        with torch.no_grad():
            out = model(x)

            # Also record intermediate routing values for sub-module testing
            x_flat = x.view(-1, cfg.hidden_size).float()
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

    # ── Collect all model weights ─────────────────────────────────────────
    weights = {}

    # Router weights
    weights["gate.weight"] = t2l(model.gate.weight)
    weights["gate.e_score_correction_bias"] = t2l(
        model.gate.e_score_correction_bias)

    # Routed expert weights
    for i, exp in enumerate(model.experts):
        weights[f"experts.{i}.gate_proj.weight"] = t2l(exp.gate_proj.weight)
        weights[f"experts.{i}.up_proj.weight"]   = t2l(exp.up_proj.weight)
        weights[f"experts.{i}.down_proj.weight"] = t2l(exp.down_proj.weight)

    # Shared expert weights
    weights["shared_experts.gate_proj.weight"] = t2l(
        model.shared_experts.gate_proj.weight)
    weights["shared_experts.up_proj.weight"]   = t2l(
        model.shared_experts.up_proj.weight)
    weights["shared_experts.down_proj.weight"] = t2l(
        model.shared_experts.down_proj.weight)

    # ── Write JSON ────────────────────────────────────────────────────────
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
        "weights":    weights,
        "test_cases": cases,
    }

    with open("test_cases.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nGenerated {n_cases} test cases → test_cases.json")


if __name__ == "__main__":
    print("Generating DeepSeek-V3 MoE test cases...\n")
    generate(n_cases=6)