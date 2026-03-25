"""
Week 5 Assignment: FlashAttention Algorithm 1 reimplemented using CuTe.

Key change from Week 4:
  Week 4: manual index arithmetic   S[r * Bc + c]
  Week 5: CuTe Layout abstraction   s_tile(r, c)

A CuTe Layout = (Shape, Stride) pair that maps coordinates to memory indices.
  Shape  = (Br, Bc)  — the coordinate space
  Stride = (Bc,  1)  — how each dimension steps through memory
  → s_tile(r, c) automatically computes r*Bc + c*1

"""

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /cutlass"
    )
)

app = modal.App("flash-attention-cute", image=image)

# ── CuTe FlashAttention kernel ────────────────────────────────────────────────
CUTE_SOURCE = r"""
/*
 * FlashAttention Algorithm 1 — CuTe Implementation (Week 5)
 *
 * The ONLY difference from Week 4 is how we index shared memory tiles.
 *
 * Week 4 (manual):              Week 5 (CuTe):
 *   S[r * Bc + c]         →      s_tile(r, c)
 *   P[r * Bc + c]         →      s_tile(r, c)   (reused)
 *   acc[r * d + k]        →      acc_tile(r, k)
 *   Q[global_row * d + k] →      q_tile(r, k)
 *   K[global_col * d + k] →      k_tile(c, k)
 *   V[global_col * d + k] →      v_tile(c, k)
 *
 * CuTe Layout recap:
 *   make_layout(make_shape(Br, Bc), make_stride(Bc, 1))
 *   → element (r, c) lives at memory offset r*Bc + c*1
 *   This is identical to the manual formula, but CuTe computes it for you.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CuTe headers from CUTLASS
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
using namespace cute;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ── Kernel ────────────────────────────────────────────────────────────────────
// Grid : (Tr,)  — one thread block per Q row-tile
// Block: (256,) — threads cooperate within a tile
template<int Br, int Bc, int D>
__global__ void flash_attention_cute_kernel(
    const float* __restrict__ Q,   // [N, D]
    const float* __restrict__ K,   // [N, D]
    const float* __restrict__ V,   // [N, D]
    float*       __restrict__ O,   // [N, D]
    float*       __restrict__ L,   // [N]
    int N)
{
    const float scale = 1.0f / sqrtf((float)D);

    int tile_i    = blockIdx.x;
    int row_start = tile_i * Br;
    int row_end   = min(row_start + Br, N);
    int br        = row_end - row_start;   // actual rows in this tile

    if (row_start >= N) return;

    int tid         = threadIdx.x;
    int num_threads = blockDim.x;
    int Tc          = (N + Bc - 1) / Bc;

    // ── Shared memory ─────────────────────────────────────────────────────
    extern __shared__ float smem[];
    float *smem_Q   = smem;
    float *smem_K   = smem_Q + Br * D;
    float *smem_V   = smem_K + Bc * D;
    float *smem_S   = smem_V + Bc * D;
    float *smem_acc = smem_S + Br * Bc;
    float *smem_mi  = smem_acc + Br * D;
    float *smem_li  = smem_mi  + Br;

    // ── CuTe: define Layouts for each tile ───────────────────────────────
    //
    // make_layout(Shape, Stride):
    //   Shape  = dimensions of the coordinate space
    //   Stride = how each dimension maps to a 1-D memory offset
    //
    // Row-major [Br, D]:  stride = (D, 1)  → element(r,k) = r*D + k
    // Row-major [Bc, D]:  stride = (D, 1)  → element(c,k) = c*D + k
    // Row-major [Br, Bc]: stride = (Bc,1)  → element(r,c) = r*Bc + c
    //
    // make_tensor(ptr, layout) wraps the raw pointer with the layout,
    // so tensor(i,j) = ptr[ layout(i,j) ] automatically.

    auto q_tile   = make_tensor(make_smem_ptr(smem_Q),
                        make_layout(make_shape(Br, D),
                                    make_stride(D,  1)));   // [Br, D]

    auto k_tile   = make_tensor(make_smem_ptr(smem_K),
                        make_layout(make_shape(Bc, D),
                                    make_stride(D,  1)));   // [Bc, D]

    auto v_tile   = make_tensor(make_smem_ptr(smem_V),
                        make_layout(make_shape(Bc, D),
                                    make_stride(D,  1)));   // [Bc, D]

    auto s_tile   = make_tensor(make_smem_ptr(smem_S),
                        make_layout(make_shape(Br, Bc),
                                    make_stride(Bc, 1)));   // [Br, Bc]

    auto acc_tile = make_tensor(make_smem_ptr(smem_acc),
                        make_layout(make_shape(Br, D),
                                    make_stride(D,  1)));   // [Br, D]

    // ── Initialise accumulators ───────────────────────────────────────────
    for (int r = tid; r < br; r += num_threads) {
        smem_mi[r] = -FLT_MAX;
        smem_li[r] = 0.0f;
    }
    for (int idx = tid; idx < br * D; idx += num_threads) {
        // Week 4: acc[idx] = 0
        // Week 5: acc_tile element — same result, CuTe handles addressing
        smem_acc[idx] = 0.0f;
    }
    __syncthreads();

    // ── Load Q tile ───────────────────────────────────────────────────────
    for (int idx = tid; idx < br * D; idx += num_threads) {
        int r = idx / D, k = idx % D;
        int global_row = row_start + r;
        // Week 4: smem_Q[r * D + k] = Q[global_row * D + k]
        // Week 5: q_tile(r, k)      = Q[global_row * D + k]
        q_tile(r, k) = (global_row < N) ? Q[global_row * D + k] : 0.0f;
    }
    __syncthreads();

    // ── Inner loop over KV tiles ──────────────────────────────────────────
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int col_end   = min(col_start + Bc, N);
        int bc        = col_end - col_start;

        // Load K_j and V_j
        for (int idx = tid; idx < bc * D; idx += num_threads) {
            int c = idx / D, k = idx % D;
            int global_col = col_start + c;
            // Week 4: smem_K[c * D + k] = K[global_col * D + k]
            // Week 5: k_tile(c, k)      = K[global_col * D + k]
            k_tile(c, k) = K[global_col * D + k];
            v_tile(c, k) = V[global_col * D + k];
        }
        __syncthreads();

        // Compute S = Q_i @ K_j^T * scale
        for (int idx = tid; idx < br * bc; idx += num_threads) {
            int r = idx / bc, c = idx % bc;
            float dot = 0.0f;
            for (int k = 0; k < D; k++) {
                // Week 4: smem_Q[r*D+k] * smem_K[c*D+k]
                // Week 5: q_tile(r,k)   * k_tile(c,k)
                dot += q_tile(r, k) * k_tile(c, k);
            }
            // Week 4: S[r * Bc + c] = dot * scale
            // Week 5: s_tile(r, c)  = dot * scale
            s_tile(r, c) = dot * scale;
        }
        __syncthreads();

        // Online softmax update (each thread handles one row)
        for (int r = tid; r < br; r += num_threads) {
            // Find row max of S
            float mij = -FLT_MAX;
            for (int c = 0; c < bc; c++) {
                // Week 4: S[r * Bc + c]
                // Week 5: s_tile(r, c)
                if (s_tile(r, c) > mij) mij = s_tile(r, c);
            }
            float m_new = fmaxf(smem_mi[r], mij);

            // Compute P = exp(S - mij), l_ij = rowsum(P)
            float lij = 0.0f;
            for (int c = 0; c < bc; c++) {
                float p = expf(s_tile(r, c) - mij);
                // Reuse s_tile to store P (same as Week 4 reusing S/P arrays)
                s_tile(r, c) = p;
                lij += p;
            }

            float alpha = expf(smem_mi[r] - m_new);
            float beta  = expf(mij        - m_new);

            // Rescale accumulator
            for (int k = 0; k < D; k++) {
                // Week 4: acc[r * d + k] *= alpha
                // Week 5: acc_tile(r, k) *= alpha
                acc_tile(r, k) *= alpha;
            }

            // Update running stats
            smem_li[r] = alpha * smem_li[r] + beta * lij;
            smem_mi[r] = m_new;

            // Accumulate P @ V_j
            for (int k = 0; k < D; k++) {
                float pv = 0.0f;
                for (int c = 0; c < bc; c++) {
                    // Week 4: P[r*Bc+c] * smem_V[c*D+k]
                    // Week 5: s_tile(r,c) * v_tile(c,k)
                    pv += s_tile(r, c) * v_tile(c, k);
                }
                acc_tile(r, k) += beta * pv;
            }
        }
        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────
    for (int idx = tid; idx < br * D; idx += num_threads) {
        int r = idx / D, k = idx % D;
        int global_row = row_start + r;
        if (global_row < N) {
            // Week 4: O[global_row*d+k] = acc[r*d+k] / l_i[r]
            // Week 5: O[global_row*D+k] = acc_tile(r,k) / smem_li[r]
            O[global_row * D + k] = acc_tile(r, k) / smem_li[r];
        }
    }
    for (int r = tid; r < br; r += num_threads) {
        int global_row = row_start + r;
        if (global_row < N)
            L[global_row] = smem_mi[r] + logf(smem_li[r]);
    }
}

// ── Host wrapper ──────────────────────────────────────────────────────────────
#define BR 32
#define BC 32
#define HD 64   // head dimension (compile-time for template)

void flash_attention_cute(
    const float *Q, const float *K, const float *V,
    float *O, float *L, int N)
{
    int Tr = (N + BR - 1) / BR;
    // smem: Q[Br,D] + K[Bc,D] + V[Bc,D] + S[Br,Bc] + acc[Br,D] + mi[Br] + li[Br]
    size_t smem = ((BR + BC + BC) * HD + BR * BC + BR * HD + BR + BR)
                  * sizeof(float);

    flash_attention_cute_kernel<BR, BC, HD>
        <<<Tr, 256, smem>>>(Q, K, V, O, L, N);
    cudaDeviceSynchronize();
}

// ── CPU reference ─────────────────────────────────────────────────────────────
void standard_attention_cpu(
    const float *Q, const float *K, const float *V,
    float *O, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    float *S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = s * scale;
        }
    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX, sum = 0;
        for (int j = 0; j < N; j++) if (S[i*N+j] > m) m = S[i*N+j];
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-m); sum += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= sum;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float s = 0;
            for (int j = 0; j < N; j++) s += S[i*N+j] * V[j*d+k];
            O[i*d+k] = s;
        }
    free(S);
}

void init_random(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    printf("FlashAttention with CuTe Layout Algebra (Week 5)\n");
    printf("=================================================\n\n");

    const int N = 512, D = HD;
    printf("Config: N=%d D=%d Br=%d Bc=%d\n\n", N, D, BR, BC);

    float *h_Q = (float*)malloc(N*D*sizeof(float));
    float *h_K = (float*)malloc(N*D*sizeof(float));
    float *h_V = (float*)malloc(N*D*sizeof(float));
    float *h_O = (float*)malloc(N*D*sizeof(float));
    float *h_L = (float*)malloc(N*sizeof(float));
    float *h_ref = (float*)malloc(N*D*sizeof(float));

    srand(42);
    init_random(h_Q, N*D);
    init_random(h_K, N*D);
    init_random(h_V, N*D);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, N*D*sizeof(float));
    cudaMalloc(&d_K, N*D*sizeof(float));
    cudaMalloc(&d_V, N*D*sizeof(float));
    cudaMalloc(&d_O, N*D*sizeof(float));
    cudaMalloc(&d_L, N*sizeof(float));

    cudaMemcpy(d_Q, h_Q, N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N*D*sizeof(float), cudaMemcpyHostToDevice);

    printf("Running CuTe FlashAttention kernel...\n");
    flash_attention_cute(d_Q, d_K, d_V, d_O, d_L, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    flash_attention_cute(d_Q, d_K, d_V, d_O, d_L, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("Done. Time: %.3f ms\n\n", ms);

    cudaMemcpy(h_O, d_O, N*D*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Running CPU reference...\n");
    standard_attention_cpu(h_Q, h_K, h_V, h_ref, N, D);
    printf("Done.\n\n");

    printf("Verifying correctness...\n");
    float max_err = 0, sum_err = 0;
    int err_count = 0;
    for (int i = 0; i < N*D; i++) {
        float e = fabsf(h_O[i] - h_ref[i]);
        if (e > max_err) max_err = e;
        sum_err += e;
        if (e > 1e-3f) err_count++;
    }
    printf("  Max error:    %.6e\n", max_err);
    printf("  Avg error:    %.6e\n", sum_err/(N*D));
    printf("  Large errors: %d / %d\n\n", err_count, N*D);
    printf(max_err < 1e-3f ? "✓ Test PASSED!\n" : "✗ Test FAILED!\n");

    free(h_Q); free(h_K); free(h_V);
    free(h_O); free(h_L); free(h_ref);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    return 0;
}
"""

# ── Modal function ────────────────────────────────────────────────────────────
@app.function(gpu="A10G", image=image, timeout=600)
def run_flash_attention():
    import subprocess

    with open("/tmp/flash_attention_cute.cu", "w") as f:
        f.write(CUTE_SOURCE)

    print("Compiling CuTe FlashAttention kernel...")
    result = subprocess.run(
        [
            "nvcc", "-O2", "-arch=sm_86",
            "--std=c++17",
            "-I/cutlass/include",
            "-I/cutlass/tools/util/include",
            "/tmp/flash_attention_cute.cu",
            "-o", "/tmp/flash_attention_cute",
            "-lm",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Compilation FAILED!")
        print(result.stderr[-3000:])
        return
    print("Compilation successful!\n")
    print("=" * 60)

    run = subprocess.run(
        ["/tmp/flash_attention_cute"],
        capture_output=True, text=True,
    )
    print(run.stdout)
    if run.stderr:
        print("STDERR:", run.stderr[-1000:])


@app.local_entrypoint()
def main():
    run_flash_attention.remote()