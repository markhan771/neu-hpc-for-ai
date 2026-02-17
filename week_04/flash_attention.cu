/*
 * FlashAttention-2 Algorithm 1 (Section 3.1) - Parallelized CUDA Implementation
 *
 * This implements the forward pass of FlashAttention using tiled computation
 * with online softmax normalization, parallelized across GPU thread blocks.
 *
 * Parallelization Strategy:
 * - Each thread block processes one row block (Br rows) independently
 * - Within each block, threads cooperate to compute the attention for those rows
 * - Different blocks can process different row tiles in parallel
 *
 * Reference: FlashAttention-2: Faster Attention with Better Parallelism
 *           https://arxiv.org/abs/2307.08691
 */

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

// Matrix access macro (row-major)
#define MAT(ptr, row, col, stride) ((ptr)[(row) * (stride) + (col)])

/*
 * FlashAttention Forward Pass Kernel
 *
 * Each thread block processes one row block (Br consecutive rows)
 * Block ID determines which row block to process
 *
 * Grid dimension: (Tr, 1, 1) where Tr = ceil(N / Br)
 * Block dimension: (THREADS_PER_BLOCK, 1, 1)
 */
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,    // N x d
    const float* __restrict__ K,    // N x d
    const float* __restrict__ V,    // N x d
    float* __restrict__ O,          // N x d (output)
    float* __restrict__ L,          // N (logsumexp output)
    const int N,                    // Sequence length
    const int d,                    // Head dimension
    const int Br,                   // Row block size
    const int Bc                    // Column block size
) {
    const float scale = 1.0f / sqrtf((float)d);

    // This block processes row block i
    const int block_row = blockIdx.x;
    const int row_start = block_row * Br;
    const int row_end = min(row_start + Br, N);
    const int rows_in_block = row_end - row_start;

    if (row_start >= N) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory for statistics and temporary matrices
    extern __shared__ float shared_mem[];

    // Layout of shared memory:
    // m_i[Br], l_i[Br], acc[Br * d], S[Br * Bc], P[Br * Bc]
    float *m_i = shared_mem;                                    // Br
    float *l_i = m_i + Br;                                      // Br
    float *acc = l_i + Br;                                      // Br * d
    float *S = acc + Br * d;                                    // Br * Bc
    float *P = S + Br * Bc;                                     // Br * Bc

    // Initialize m_i and l_i
    for (int r = tid; r < rows_in_block; r += num_threads) {
        m_i[r] = -FLT_MAX;
        l_i[r] = 0.0f;
    }

    // Initialize acc to zero
    for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
        acc[idx] = 0.0f;
    }
    __syncthreads();

    // Number of column blocks
    const int Tc = (N + Bc - 1) / Bc;

    // Process each column block
    for (int j = 0; j < Tc; j++) {
        const int col_start = j * Bc;
        const int col_end = min(col_start + Bc, N);
        const int cols_in_block = col_end - col_start;

        // Step 1: Compute S = Q_i @ K_j^T * scale
        // Each thread computes multiple elements
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

        // Step 2: Compute m_ij = max(m_i, rowmax(S))
        // Each thread processes one or more rows
        for (int r = tid; r < rows_in_block; r += num_threads) {
            float row_max = -FLT_MAX;
            for (int c = 0; c < cols_in_block; c++) {
                float val = S[r * Bc + c];
                if (val > row_max) row_max = val;
            }
            float m_ij = fmaxf(m_i[r], row_max);

            // Step 3 & 4: Compute P = exp(S - m_ij) and l_ij = rowsum(P)
            float l_ij = 0.0f;
            for (int c = 0; c < cols_in_block; c++) {
                float p_val = expf(S[r * Bc + c] - m_ij);
                P[r * Bc + c] = p_val;
                l_ij += p_val;
            }

            // Step 5: Compute alpha = exp(m_i - m_ij)
            float alpha = expf(m_i[r] - m_ij);

            // Step 6: Update acc = acc * alpha (for this row)
            for (int k = 0; k < d; k++) {
                acc[r * d + k] *= alpha;
            }

            // Step 7: Update l_i = l_i * alpha + l_ij
            l_i[r] = l_i[r] * alpha + l_ij;

            // Step 8: Update m_i = m_ij
            m_i[r] = m_ij;
        }
        __syncthreads();

        // Step 6 (continued): Add P @ V_j to acc
        // Each thread computes multiple elements of the result
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

    // Step 9: Final normalization O_i = acc / l_i and compute logsumexp
    for (int idx = tid; idx < rows_in_block * d; idx += num_threads) {
        const int r = idx / d;
        const int k = idx % d;
        MAT(O, row_start + r, k, d) = acc[r * d + k] / l_i[r];
    }

    // Write logsumexp: L = m_i + log(l_i)
    for (int r = tid; r < rows_in_block; r += num_threads) {
        L[row_start + r] = m_i[r] + logf(l_i[r]);
    }
}

// Host function to launch kernel
void flash_attention_forward_cuda(
    const float *Q,    // N x d (device)
    const float *K,    // N x d (device)
    const float *V,    // N x d (device)
    float *O,          // N x d (device)
    float *L,          // N (device)
    int N,             // Sequence length
    int d,             // Head dimension
    int Br,            // Row block size
    int Bc             // Column block size
) {
    // Grid dimension: one block per row tile
    const int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, 1, 1);

    // Block dimension: use 256 threads per block (typical choice)
    const int threads_per_block = 256;
    dim3 block(threads_per_block, 1, 1);

    // Shared memory size calculation
    // m_i[Br] + l_i[Br] + acc[Br*d] + S[Br*Bc] + P[Br*Bc]
    size_t shared_mem_size = (Br + Br + Br * d + Br * Bc + Br * Bc) * sizeof(float);

    // Launch kernel
    flash_attention_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, O, L, N, d, Br, Bc
    );

    CUDA_CHECK(cudaGetLastError());
}

// Reference implementation on CPU for verification
void standard_attention_cpu(
    const float *Q,    // N x d
    const float *K,    // N x d
    const float *V,    // N x d
    float *O,          // N x d
    int N,
    int d
) {
    const float scale = 1.0f / sqrtf((float)d);

    // Allocate S = Q @ K^T
    float *S = (float*)malloc(N * N * sizeof(float));

    // Compute S = Q @ K^T * scale
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += MAT(Q, i, k, d) * MAT(K, j, k, d);
            }
            MAT(S, i, j, N) = sum * scale;
        }
    }

    // Apply softmax row-wise
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            if (MAT(S, i, j, N) > row_max) row_max = MAT(S, i, j, N);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            MAT(S, i, j, N) = expf(MAT(S, i, j, N) - row_max);
            sum += MAT(S, i, j, N);
        }

        // Normalize
        for (int j = 0; j < N; j++) {
            MAT(S, i, j, N) /= sum;
        }
    }

    // Compute O = S @ V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += MAT(S, i, j, N) * MAT(V, j, k, d);
            }
            MAT(O, i, k, d) = sum;
        }
    }

    free(S);
}

// Helper function: Initialize matrix with random values
void init_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Test program
int main() {
    printf("FlashAttention-2 CUDA Forward Pass Test\n");
    printf("========================================\n\n");

    // Parameters
    const int N = 512;   // Sequence length
    const int d = 64;    // Head dimension
    const int Br = 64;   // Row block size
    const int Bc = 64;   // Column block size

    printf("Configuration:\n");
    printf("  Sequence length (N): %d\n", N);
    printf("  Head dimension (d):  %d\n", d);
    printf("  Row block size (Br): %d\n", Br);
    printf("  Col block size (Bc): %d\n\n", Bc);

    // Allocate host memory
    float *h_Q = (float*)malloc(N * d * sizeof(float));
    float *h_K = (float*)malloc(N * d * sizeof(float));
    float *h_V = (float*)malloc(N * d * sizeof(float));
    float *h_O_cuda = (float*)malloc(N * d * sizeof(float));
    float *h_O_ref = (float*)malloc(N * d * sizeof(float));
    float *h_L = (float*)malloc(N * sizeof(float));

    // Initialize with random values
    srand(42);
    init_random(h_Q, N * d);
    init_random(h_K, N * d);
    init_random(h_V, N * d);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L, N * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N * d * sizeof(float), cudaMemcpyHostToDevice));

    // Run CUDA kernel
    printf("Running FlashAttention CUDA kernel...\n");

    // Warm-up run
    flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Done. Time: %.3f ms\n\n", milliseconds);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O_cuda, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));

    // Run reference implementation
    printf("Running CPU reference implementation...\n");
    standard_attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);
    printf("Done.\n\n");

    // Verify correctness
    printf("Verifying correctness...\n");
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < N * d; i++) {
        float error = fabsf(h_O_cuda[i] - h_O_ref[i]);
        if (error > max_error) max_error = error;
        sum_error += error;
        if (error > 1e-3f) error_count++;
    }
    float avg_error = sum_error / (N * d);

    printf("  Max error:       %.6e\n", max_error);
    printf("  Avg error:       %.6e\n", avg_error);
    printf("  Large errors:    %d / %d\n\n", error_count, N * d);

    if (max_error < 1e-3f) {
        printf("✓ Test PASSED!\n");
    } else {
        printf("✗ Test FAILED (errors exceed threshold)\n");
        printf("  Note: Small numerical differences are expected due to floating point precision\n");
    }

    // Cleanup
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_cuda);
    free(h_O_ref);
    free(h_L);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
