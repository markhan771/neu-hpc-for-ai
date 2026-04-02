"""
Modal script to compile and run FlashAttention CUDA implementation on cloud GPU.
The CUDA source is embedded directly in this file — no separate .cu file needed.
Usage:
    modal run modal_flash_attention.py
"""
import modal

app = modal.App("flash-attention")

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
)

# ── CUDA source embedded directly ────────────────────────────────────────────
CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define MAT(ptr, row, col, stride) ((ptr)[(row) * (stride) + (col)])

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int N,
    const int d,
    const int Br,
    const int Bc
) {
    const float scale = 1.0f / sqrtf((float)d);

    const int block_row = blockIdx.x;
    const int row_start = block_row * Br;
    const int row_end = min(row_start + Br, N);
    const int rows_in_block = row_end - row_start;

    if (row_start >= N) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float shared_mem[];
    float *m_i = shared_mem;
    float *l_i = m_i + Br;
    float *acc = l_i + Br;
    float *S   = acc + Br * d;
    float *P   = S   + Br * Bc;

    for (int r = tid; r < rows_in_block; r += num_threads) {
        m_i[r] = -FLT_MAX;
        l_i[r] = 0.0f;
    }
    for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
        acc[idx] = 0.0f;
    }
    __syncthreads();

    const int Tc = (N + Bc - 1) / Bc;

    for (int j = 0; j < Tc; j++) {
        const int col_start = j * Bc;
        const int col_end = min(col_start + Bc, N);
        const int cols_in_block = col_end - col_start;

        for (int idx = tid; idx < rows_in_block * cols_in_block; idx += num_threads) {
            const int r = idx / cols_in_block;
            const int c = idx % cols_in_block;
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += MAT(Q, row_start + r, k, d) * MAT(K, col_start + c, k, d);
            }
            S[r * Bc + c] = sum * scale;
        }
        __syncthreads();

        for (int r = tid; r < rows_in_block; r += num_threads) {
            float row_max = -FLT_MAX;
            for (int c = 0; c < cols_in_block; c++) {
                float val = S[r * Bc + c];
                if (val > row_max) row_max = val;
            }
            float m_ij = fmaxf(m_i[r], row_max);

            float l_ij = 0.0f;
            for (int c = 0; c < cols_in_block; c++) {
                float p_val = expf(S[r * Bc + c] - m_ij);
                P[r * Bc + c] = p_val;
                l_ij += p_val;
            }

            float alpha = expf(m_i[r] - m_ij);
            for (int k = 0; k < d; k++) {
                acc[r * d + k] *= alpha;
            }
            l_i[r] = l_i[r] * alpha + l_ij;
            m_i[r] = m_ij;
        }
        __syncthreads();

        for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
            const int r = idx / d;
            const int k = idx % d;
            float sum = 0.0f;
            for (int c = 0; c < cols_in_block; c++) {
                sum += P[r * Bc + c] * MAT(V, col_start + c, k, d);
            }
            acc[r * d + k] += sum;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
        const int r = idx / d;
        const int k = idx % d;
        MAT(O, row_start + r, k, d) = acc[r * d + k] / l_i[r];
    }
    for (int r = tid; r < rows_in_block; r += num_threads) {
        L[row_start + r] = m_i[r] + logf(l_i[r]);
    }
}

void flash_attention_forward_cuda(
    const float *Q, const float *K, const float *V,
    float *O, float *L,
    int N, int d, int Br, int Bc
) {
    const int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, 1, 1);
    dim3 block(256, 1, 1);
    size_t shared_mem_size = (Br + Br + Br*d + Br*Bc + Br*Bc) * sizeof(float);
    flash_attention_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, O, L, N, d, Br, Bc);
    CUDA_CHECK(cudaGetLastError());
}

void standard_attention_cpu(
    const float *Q, const float *K, const float *V,
    float *O, int N, int d
) {
    const float scale = 1.0f / sqrtf((float)d);
    float *S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += MAT(Q,i,k,d) * MAT(K,j,k,d);
            MAT(S,i,j,N) = sum * scale;
        }
    for (int i = 0; i < N; i++) {
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++)
            if (MAT(S,i,j,N) > row_max) row_max = MAT(S,i,j,N);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            MAT(S,i,j,N) = expf(MAT(S,i,j,N) - row_max);
            sum += MAT(S,i,j,N);
        }
        for (int j = 0; j < N; j++) MAT(S,i,j,N) /= sum;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
                sum += MAT(S,i,j,N) * MAT(V,j,k,d);
            MAT(O,i,k,d) = sum;
        }
    free(S);
}

void init_random(float *data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    printf("FlashAttention-2 CUDA Forward Pass Test\n");
    printf("========================================\n\n");

    const int N = 512, d = 64, Br = 32, Bc = 32;
    printf("Configuration:\n");
    printf("  Sequence length (N): %d\n", N);
    printf("  Head dimension (d):  %d\n", d);
    printf("  Row block size (Br): %d\n", Br);
    printf("  Col block size (Bc): %d\n\n", Bc);
    float *h_Q = (float*)malloc(N*d*sizeof(float));
    float *h_K = (float*)malloc(N*d*sizeof(float));
    float *h_V = (float*)malloc(N*d*sizeof(float));
    float *h_O_cuda = (float*)malloc(N*d*sizeof(float));
    float *h_O_ref  = (float*)malloc(N*d*sizeof(float));
    float *h_L = (float*)malloc(N*sizeof(float));

    srand(42);
    init_random(h_Q, N*d);
    init_random(h_K, N*d);
    init_random(h_V, N*d);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L, N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N*d*sizeof(float), cudaMemcpyHostToDevice));

    printf("Running FlashAttention CUDA kernel...\n");
    flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Done. Time: %.3f ms\n\n", ms);

    CUDA_CHECK(cudaMemcpy(h_O_cuda, d_O, N*d*sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference implementation...\n");
    standard_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);
    printf("Done.\n\n");

    printf("Verifying correctness...\n");
    float max_error = 0.0f, sum_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < N*d; i++) {
        float error = fabsf(h_O_cuda[i] - h_O_ref[i]);
        if (error > max_error) max_error = error;
        sum_error += error;
        if (error > 1e-3f) error_count++;
    }
    printf("  Max error:    %.6e\n", max_error);
    printf("  Avg error:    %.6e\n", sum_error / (N*d));
    printf("  Large errors: %d / %d\n\n", error_count, N*d);
    printf(max_error < 1e-3f ? "✓ Test PASSED!\n" : "✗ Test FAILED!\n");

    free(h_Q); free(h_K); free(h_V);
    free(h_O_cuda); free(h_O_ref); free(h_L);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    return 0;
}
"""

# ── Modal function ────────────────────────────────────────────────────────────
@app.function(gpu="T4", image=cuda_image, timeout=600)
def run_flash_attention():
    import subprocess

    # Write embedded CUDA source to temp file
    with open("/tmp/flash_attention.cu", "w") as f:
        f.write(CUDA_SOURCE)

    print("Compiling CUDA code...")
    result = subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-O3", "-arch=sm_75",
            "/tmp/flash_attention.cu",
            "-o", "/tmp/flash_attention_cuda",
            "-lm",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Compilation FAILED!")
        print(result.stderr)
        return
    print("Compilation successful!\n")
    print("=" * 60)

    run = subprocess.run(
        ["/tmp/flash_attention_cuda"],
        capture_output=True, text=True,
    )
    print(run.stdout)
    if run.stderr:
        print("STDERR:", run.stderr)


@app.local_entrypoint()
def main():
    run_flash_attention.remote()