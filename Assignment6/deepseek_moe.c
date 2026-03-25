/*
 * DeepSeek-V3 MoE Operator — Pure C Implementation (Week 7)
 *
 * No CUDA, no parallelism. Single-threaded CPU reference.
 *
 * Reads weights and test cases from test_cases.json,
 * runs the MoE forward pass, and checks outputs against
 * the PyTorch reference values.
 *
 * Forward pass:
 *   output = input
 *          + sum_k  gate_k * routed_expert_k(input)
 *          + shared_expert(input)
 *
 * Each expert FFN (SwiGLU):
 *   h      = silu(gate_proj(x)) * up_proj(x)
 *   output = down_proj(h)
 *
 * Router (noaux_tc grouped top-K):
 *   scores    = sigmoid(x @ W_gate^T)
 *   sfc       = scores + correction_bias
 *   group top-K → select topk_group groups
 *   global top-K within selected groups → topk_idx
 *   gate weights = scores[topk_idx], normalised + scaled
 *
 * Compile:
 *   gcc -O2 -o deepseek_moe deepseek_moe.c -lm
 * Run:
 *   ./deepseek_moe test_cases.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── tiny JSON parser (numbers only, no dependencies) ──────────────────────── */
/* We implement a minimal recursive-descent parser sufficient for our JSON. */

typedef struct { const char *p; } Parser;

static void skip_ws(Parser *p) {
    while (*p->p == ' ' || *p->p == '\n' ||
           *p->p == '\r' || *p->p == '\t') p->p++;
}

static void expect(Parser *p, char c) {
    skip_ws(p);
    if (*p->p != c) {
        fprintf(stderr, "JSON parse error: expected '%c', got '%c'\n", c, *p->p);
        exit(1);
    }
    p->p++;
}

static double parse_number(Parser *p) {
    skip_ws(p);
    char *end;
    double v = strtod(p->p, &end);
    p->p = end;
    return v;
}

/* Parse any JSON array of numbers (flat or nested up to any depth)
 * into a pre-allocated float buffer. Returns the number of elements read. */
static int parse_any_array(Parser *p, float *buf, int max_len) {
    expect(p, '[');
    int total = 0;
    skip_ws(p);
    if (*p->p == ']') { p->p++; return 0; }
    while (1) {
        skip_ws(p);
        if (*p->p == '[') {
            /* nested array — recurse */
            total += parse_any_array(p, buf + total, max_len - total);
        } else {
            if (total >= max_len) { fprintf(stderr, "Array overflow\n"); exit(1); }
            buf[total++] = (float)parse_number(p);
        }
        skip_ws(p);
        if (*p->p == ']') { p->p++; break; }
        expect(p, ',');
    }
    return total;
}

/* Aliases so the rest of the code compiles unchanged */
static int parse_float_array(Parser *p, float *buf, int max_len) {
    return parse_any_array(p, buf, max_len);
}
static int parse_nested_float_array(Parser *p, float *buf, int max_len) {
    return parse_any_array(p, buf, max_len);
}

/* Advance parser past a JSON string key (including quotes). */
static void skip_string(Parser *p) {
    expect(p, '"');
    while (*p->p && *p->p != '"') {
        if (*p->p == '\\') p->p++;   /* escaped character */
        p->p++;
    }
    expect(p, '"');
}

/* Return 1 if the next token is the string key `key`, consume it + ':'. */
static int match_key(Parser *p, const char *key) {
    skip_ws(p);
    if (*p->p != '"') return 0;
    const char *save = p->p;
    p->p++;                          /* skip opening quote */
    size_t klen = strlen(key);
    if (strncmp(p->p, key, klen) == 0 && p->p[klen] == '"') {
        p->p += klen + 1;            /* skip key + closing quote */
        skip_ws(p);
        expect(p, ':');
        return 1;
    }
    p->p = save;                     /* restore */
    return 0;
}

/* Skip any JSON value (object, array, string, number, bool, null). */
static void skip_value(Parser *p);
static void skip_object(Parser *p) {
    expect(p, '{');
    skip_ws(p);
    if (*p->p == '}') { p->p++; return; }
    while (1) {
        skip_string(p);
        skip_ws(p); expect(p, ':');
        skip_value(p);
        skip_ws(p);
        if (*p->p == '}') { p->p++; break; }
        expect(p, ',');
    }
}
static void skip_array_v(Parser *p) {
    expect(p, '[');
    skip_ws(p);
    if (*p->p == ']') { p->p++; return; }
    while (1) {
        skip_value(p);
        skip_ws(p);
        if (*p->p == ']') { p->p++; break; }
        expect(p, ',');
    }
}
static void skip_value(Parser *p) {
    skip_ws(p);
    if (*p->p == '{')      skip_object(p);
    else if (*p->p == '[') skip_array_v(p);
    else if (*p->p == '"') skip_string(p);
    else {
        char *end; strtod(p->p, &end);
        if (end == p->p) {
            /* bool / null */
            while (*p->p && *p->p != ',' && *p->p != '}' && *p->p != ']')
                p->p++;
        } else p->p = end;
    }
}

/* ── model dimensions ───────────────────────────────────────────────────────── */
typedef struct {
    int H;          /* hidden_size */
    int I;          /* moe_intermediate_size */
    int E;          /* n_routed_experts */
    int K;          /* num_experts_per_tok */
    int Ns;         /* n_shared_experts */
    int n_group;
    int topk_group;
    float routed_scaling_factor;
    int norm_topk_prob;
} Config;

/* ── weight storage ─────────────────────────────────────────────────────────── */
typedef struct {
    /* Router */
    float *gate_W;      /* [E, H] */
    float *gate_bias;   /* [E]    */
    /* Routed experts: each expert has gate/up/down weight matrices */
    float **exp_gate;   /* [E][I*H] */
    float **exp_up;     /* [E][I*H] */
    float **exp_down;   /* [E][H*I] */
    /* Shared expert */
    float *sh_gate;     /* [I*Ns, H] → shape (I*Ns, H) */
    float *sh_up;
    float *sh_down;
} Weights;

/* ── math helpers ───────────────────────────────────────────────────────────── */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Matrix-vector product: y = W x,  W is [out, in] row-major */
static void matvec(const float *W, const float *x, float *y,
                   int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < in_dim; j++)
            s += W[i * in_dim + j] * x[j];
        y[i] = s;
    }
}

/* ── SwiGLU Expert FFN ──────────────────────────────────────────────────────── */
/*
 *  h      = silu(gate_proj(x)) * up_proj(x)   element-wise
 *  output = down_proj(h)
 *
 *  gate_proj: [inter, H]
 *  up_proj:   [inter, H]
 *  down_proj: [H, inter]
 */
static void expert_ffn(
    const float *x,         /* [H] input */
    const float *W_gate,    /* [inter, H] */
    const float *W_up,      /* [inter, H] */
    const float *W_down,    /* [H, inter] */
    float       *out,       /* [H] output */
    int H, int inter)
{
    float *g = (float *)malloc(inter * sizeof(float));
    float *u = (float *)malloc(inter * sizeof(float));
    float *h = (float *)malloc(inter * sizeof(float));

    matvec(W_gate, x, g, inter, H);
    matvec(W_up,   x, u, inter, H);

    /* h = silu(g) * u  (SwiGLU gate) */
    for (int i = 0; i < inter; i++)
        h[i] = silu(g[i]) * u[i];

    matvec(W_down, h, out, H, inter);

    free(g); free(u); free(h);
}

/* ── Router ─────────────────────────────────────────────────────────────────── */
/*
 * Returns topk_idx[K] and topk_weight[K] for a single token vector x[H].
 *
 * Algorithm:
 *   1. scores = sigmoid(W_gate @ x)
 *   2. sfc    = scores + correction_bias
 *   3. group each expert block → group_scores (top-2 sum per group)
 *   4. select topk_group groups
 *   5. mask out experts not in selected groups
 *   6. global top-K in the masked scores
 *   7. weights = scores[topk_idx], normalised, scaled
 */
static void router(
    const float *x,          /* [H] */
    const float *W_gate,     /* [E, H] */
    const float *bias,       /* [E] */
    int         *topk_idx,   /* [K] output */
    float       *topk_w,     /* [K] output */
    const Config *cfg)
{
    int E        = cfg->E;
    int K        = cfg->K;
    int H        = cfg->H;
    int n_group  = cfg->n_group;
    int tg       = cfg->topk_group;
    int E_per_g  = E / n_group;

    float *logits = (float *)malloc(E * sizeof(float));
    float *scores = (float *)malloc(E * sizeof(float));
    float *sfc    = (float *)malloc(E * sizeof(float));

    /* Step 1-2: scores and sfc */
    matvec(W_gate, x, logits, E, H);
    for (int i = 0; i < E; i++) {
        scores[i] = sigmoid(logits[i]);
        sfc[i]    = scores[i] + bias[i];
    }

    /* Step 3: group scores = sum of top-2 sfc values in each group */
    float *group_sc = (float *)malloc(n_group * sizeof(float));
    for (int g = 0; g < n_group; g++) {
        float a = -FLT_MAX, b = -FLT_MAX;
        for (int e = 0; e < E_per_g; e++) {
            float v = sfc[g * E_per_g + e];
            if (v > a)      { b = a; a = v; }
            else if (v > b) { b = v; }
        }
        group_sc[g] = a + (b == -FLT_MAX ? 0.0f : b);
    }

    /* Step 4: select top-tg groups (simple insertion sort for small tg) */
    int *sel_groups = (int *)malloc(tg * sizeof(int));
    {
        float *tmp = (float *)malloc(n_group * sizeof(float));
        memcpy(tmp, group_sc, n_group * sizeof(float));
        for (int i = 0; i < tg; i++) {
            int best = 0;
            for (int g = 1; g < n_group; g++)
                if (tmp[g] > tmp[best]) best = g;
            sel_groups[i] = best;
            tmp[best] = -FLT_MAX;
        }
        free(tmp);
    }

    /* Step 5: build mask — which experts are eligible? */
    float *masked = (float *)malloc(E * sizeof(float));
    memset(masked, 0, E * sizeof(float));
    /* fill selected groups with -inf first, then overwrite with sfc */
    for (int i = 0; i < E; i++) masked[i] = -FLT_MAX;
    for (int i = 0; i < tg; i++) {
        int g = sel_groups[i];
        for (int e = 0; e < E_per_g; e++)
            masked[g * E_per_g + e] = sfc[g * E_per_g + e];
    }

    /* Step 6: global top-K from masked scores */
    {
        float *tmp = (float *)malloc(E * sizeof(float));
        memcpy(tmp, masked, E * sizeof(float));
        for (int k = 0; k < K; k++) {
            int best = 0;
            for (int e = 1; e < E; e++)
                if (tmp[e] > tmp[best]) best = e;
            topk_idx[k] = best;
            tmp[best] = -FLT_MAX;
        }
        free(tmp);
    }

    /* Step 7: weights from original scores (no bias), normalise, scale */
    float wsum = 0.0f;
    for (int k = 0; k < K; k++) {
        topk_w[k] = scores[topk_idx[k]];
        wsum += topk_w[k];
    }
    if (K > 1 && cfg->norm_topk_prob) {
        for (int k = 0; k < K; k++)
            topk_w[k] /= (wsum + 1e-20f);
    }
    for (int k = 0; k < K; k++)
        topk_w[k] *= cfg->routed_scaling_factor;

    free(logits); free(scores); free(sfc);
    free(group_sc); free(sel_groups); free(masked);
}

/* ── Full MoE forward pass for a single token ──────────────────────────────── */
static void moe_forward_token(
    const float  *x,     /* [H] input token */
    float        *out,   /* [H] output */
    const Weights *W,
    const Config  *cfg)
{
    int H  = cfg->H;
    int I  = cfg->I;
    int K  = cfg->K;
    int Ns = cfg->Ns;

    int   *topk_idx = (int   *)malloc(K * sizeof(int));
    float *topk_w   = (float *)malloc(K * sizeof(float));
    float *tmp      = (float *)malloc(H * sizeof(float));

    /* Initialise output to input (residual connection) */
    memcpy(out, x, H * sizeof(float));

    /* Routing */
    router(x, W->gate_W, W->gate_bias, topk_idx, topk_w, cfg);

    /* Routed expert contributions */
    for (int k = 0; k < K; k++) {
        int eid = topk_idx[k];
        expert_ffn(x,
                   W->exp_gate[eid], W->exp_up[eid], W->exp_down[eid],
                   tmp, H, I);
        for (int d = 0; d < H; d++)
            out[d] += topk_w[k] * tmp[d];
    }

    /* Shared expert contribution (applied to every token) */
    int shared_inter = I * Ns;
    expert_ffn(x, W->sh_gate, W->sh_up, W->sh_down, tmp, H, shared_inter);
    for (int d = 0; d < H; d++)
        out[d] += tmp[d];

    free(topk_idx); free(topk_w); free(tmp);
}

/* ── JSON loading helpers ────────────────────────────────────────────────────── */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* Advance parser to just after the key `key` in the current object level.
 * Handles arbitrary nesting by scanning for the key string. */
static void seek_key(Parser *p, const char *key) {
    /* We search within the current object for the key.
     * Simple linear scan: skip key-value pairs until we find it. */
    while (1) {
        skip_ws(p);
        if (*p->p == '}' || *p->p == '\0') {
            fprintf(stderr, "Key '%s' not found\n", key);
            exit(1);
        }
        if (*p->p == ',') { p->p++; skip_ws(p); }
        if (match_key(p, key)) return;
        /* Key didn't match — skip the key string and its value */
        skip_ws(p);
        if (*p->p == '"') skip_string(p);
        skip_ws(p);
        if (*p->p == ':') { p->p++; skip_value(p); }
    }
}

/* ── Main ───────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s test_cases.json\n", argv[0]);
        return 1;
    }

    char   *json = read_file(argv[1]);
    Parser  pr   = { json };

    printf("DeepSeek-V3 MoE — Pure C Implementation\n");
    printf("==========================================\n\n");

    /* ── parse config ──────────────────────────────────────────────────── */
    expect(&pr, '{');
    seek_key(&pr, "config");
    expect(&pr, '{');

    Config cfg = {0};
    /* Read all config fields */
    for (int i = 0; i < 9; i++) {
        skip_ws(&pr);
        if (*pr.p == '}') break;
        if (*pr.p == ',') pr.p++;
        skip_ws(&pr);
        if      (match_key(&pr, "hidden_size"))           cfg.H    = (int)parse_number(&pr);
        else if (match_key(&pr, "moe_intermediate_size")) cfg.I    = (int)parse_number(&pr);
        else if (match_key(&pr, "n_routed_experts"))      cfg.E    = (int)parse_number(&pr);
        else if (match_key(&pr, "num_experts_per_tok"))   cfg.K    = (int)parse_number(&pr);
        else if (match_key(&pr, "n_shared_experts"))      cfg.Ns   = (int)parse_number(&pr);
        else if (match_key(&pr, "n_group"))               cfg.n_group  = (int)parse_number(&pr);
        else if (match_key(&pr, "topk_group"))            cfg.topk_group = (int)parse_number(&pr);
        else if (match_key(&pr, "routed_scaling_factor")) cfg.routed_scaling_factor = (float)parse_number(&pr);
        else if (match_key(&pr, "norm_topk_prob"))        cfg.norm_topk_prob = (int)parse_number(&pr);
        else { skip_string(&pr); skip_ws(&pr); pr.p++; skip_value(&pr); }
    }
    expect(&pr, '}');

    printf("Config: H=%d I=%d E=%d K=%d Ns=%d n_group=%d\n\n",
           cfg.H, cfg.I, cfg.E, cfg.K, cfg.Ns, cfg.n_group);

    /* ── allocate weight buffers ───────────────────────────────────────── */
    Weights W;
    W.gate_W    = (float *)malloc(cfg.E * cfg.H * sizeof(float));
    W.gate_bias = (float *)malloc(cfg.E          * sizeof(float));

    W.exp_gate = (float **)malloc(cfg.E * sizeof(float *));
    W.exp_up   = (float **)malloc(cfg.E * sizeof(float *));
    W.exp_down = (float **)malloc(cfg.E * sizeof(float *));
    for (int e = 0; e < cfg.E; e++) {
        W.exp_gate[e] = (float *)malloc(cfg.I * cfg.H * sizeof(float));
        W.exp_up[e]   = (float *)malloc(cfg.I * cfg.H * sizeof(float));
        W.exp_down[e] = (float *)malloc(cfg.H * cfg.I * sizeof(float));
    }

    int shared_inter = cfg.I * cfg.Ns;
    W.sh_gate = (float *)malloc(shared_inter * cfg.H * sizeof(float));
    W.sh_up   = (float *)malloc(shared_inter * cfg.H * sizeof(float));
    W.sh_down = (float *)malloc(cfg.H * shared_inter * sizeof(float));

    /* ── parse weights ─────────────────────────────────────────────────── */
    seek_key(&pr, "weights");
    expect(&pr, '{');

    char key_buf[64];

    /* gate.weight */
    seek_key(&pr, "gate.weight");
    parse_nested_float_array(&pr, W.gate_W, cfg.E * cfg.H);

    /* gate.e_score_correction_bias */
    skip_ws(&pr); if (*pr.p == ',') pr.p++;
    seek_key(&pr, "gate.e_score_correction_bias");
    parse_float_array(&pr, W.gate_bias, cfg.E);

    /* expert weights */
    for (int e = 0; e < cfg.E; e++) {
        skip_ws(&pr); if (*pr.p == ',') pr.p++;
        snprintf(key_buf, sizeof(key_buf), "experts.%d.gate_proj.weight", e);
        seek_key(&pr, key_buf);
        parse_nested_float_array(&pr, W.exp_gate[e], cfg.I * cfg.H);

        skip_ws(&pr); if (*pr.p == ',') pr.p++;
        snprintf(key_buf, sizeof(key_buf), "experts.%d.up_proj.weight", e);
        seek_key(&pr, key_buf);
        parse_nested_float_array(&pr, W.exp_up[e], cfg.I * cfg.H);

        skip_ws(&pr); if (*pr.p == ',') pr.p++;
        snprintf(key_buf, sizeof(key_buf), "experts.%d.down_proj.weight", e);
        seek_key(&pr, key_buf);
        parse_nested_float_array(&pr, W.exp_down[e], cfg.H * cfg.I);
    }

    /* shared expert weights */
    skip_ws(&pr); if (*pr.p == ',') pr.p++;
    seek_key(&pr, "shared_experts.gate_proj.weight");
    parse_nested_float_array(&pr, W.sh_gate, shared_inter * cfg.H);

    skip_ws(&pr); if (*pr.p == ',') pr.p++;
    seek_key(&pr, "shared_experts.up_proj.weight");
    parse_nested_float_array(&pr, W.sh_up, shared_inter * cfg.H);

    skip_ws(&pr); if (*pr.p == ',') pr.p++;
    seek_key(&pr, "shared_experts.down_proj.weight");
    parse_nested_float_array(&pr, W.sh_down, cfg.H * shared_inter);

    expect(&pr, '}');   /* close weights object */

    /* ── parse and run test cases ──────────────────────────────────────── */
    seek_key(&pr, "test_cases");
    expect(&pr, '[');

    int n_pass = 0, n_fail = 0;

    while (1) {
        skip_ws(&pr);
        if (*pr.p == ']') { pr.p++; break; }
        if (*pr.p == ',') { pr.p++; skip_ws(&pr); }

        expect(&pr, '{');

        int case_id = 0, S = 0;
        float *input = NULL, *ref_out = NULL;

        /* Read test case fields */
        for (int fi = 0; fi < 10; fi++) {
            skip_ws(&pr);
            if (*pr.p == '}') break;
            if (*pr.p == ',') pr.p++;
            skip_ws(&pr);

            if (match_key(&pr, "id")) {
                case_id = (int)parse_number(&pr);
            } else if (match_key(&pr, "S")) {
                S = (int)parse_number(&pr);
                input   = (float *)malloc(S * cfg.H * sizeof(float));
                ref_out = (float *)malloc(S * cfg.H * sizeof(float));
            } else if (match_key(&pr, "input")) {
                parse_nested_float_array(&pr, input, S * cfg.H);
            } else if (match_key(&pr, "output")) {
                parse_nested_float_array(&pr, ref_out, S * cfg.H);
            } else {
                /* skip B, H, gate_logits, gate_scores, topk_idx, topk_weight */
                skip_ws(&pr);
                if (*pr.p == '"') skip_string(&pr);
                skip_ws(&pr);
                if (*pr.p == ':') { pr.p++; skip_value(&pr); }
            }
        }
        expect(&pr, '}');

        /* Run MoE forward for each token in the sequence */
        float *my_out = (float *)malloc(S * cfg.H * sizeof(float));
        for (int t = 0; t < S; t++) {
            moe_forward_token(input   + t * cfg.H,
                              my_out  + t * cfg.H,
                              &W, &cfg);
        }

        /* Compare with reference output */
        float max_err = 0.0f;
        for (int i = 0; i < S * cfg.H; i++) {
            float e = fabsf(my_out[i] - ref_out[i]);
            if (e > max_err) max_err = e;
        }

        int pass = (max_err < 1e-4f);
        printf("Case %d (S=%d): max_err=%.2e  %s\n",
               case_id, S, max_err, pass ? "PASS ✓" : "FAIL ✗");
        if (pass) n_pass++; else n_fail++;

        free(input); free(ref_out); free(my_out);
    }

    printf("\n%d / %d test cases passed.\n", n_pass, n_pass + n_fail);

    /* ── cleanup ───────────────────────────────────────────────────────── */
    free(W.gate_W); free(W.gate_bias);
    for (int e = 0; e < cfg.E; e++) {
        free(W.exp_gate[e]); free(W.exp_up[e]); free(W.exp_down[e]);
    }
    free(W.exp_gate); free(W.exp_up); free(W.exp_down);
    free(W.sh_gate); free(W.sh_up); free(W.sh_down);
    free(json);
    return (n_fail == 0) ? 0 : 1;
}