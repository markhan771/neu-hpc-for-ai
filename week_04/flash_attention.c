/*
 * FlashAttention-2 Algorithm 1 (Section 3.1) - Unparallelized C Implementation
 *
 * This implements the forward pass of FlashAttention using tiled computation
 * with online softmax normalization.
 *
 * Reference: FlashAttention-2: Faster Attention with Better Parallelism
 *           https://arxiv.org/abs/2307.08691
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// Matrix structure for convenience
typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

// Helper function: Create matrix
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float*)malloc(rows * cols * sizeof(float));
    return m;
}

// Helper function: Free matrix
void free_matrix(Matrix *m) {
    free(m->data);
    m->data = NULL;
}

// Helper function: Matrix element access
#define MAT(m, i, j) ((m).data[(i) * (m).cols + (j)])

// Helper function: Initialize matrix with random values
void init_random(Matrix m) {
    for (int i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Helper function: max of two floats (in case fmaxf is not available)
static inline float max_float(float a, float b) {
    return (a > b) ? a : b;
}

// Helper function: absolute value (in case abs_float is not available)
static inline float abs_float(float a) {
    return (a < 0.0f) ? -a : a;
}

/*
 * FlashAttention Forward Pass - Algorithm 1
 *
 * Inputs:
 *   Q: Query matrix (N x d)
 *   K: Key matrix (N x d)
 *   V: Value matrix (N x d)
 *   N: Sequence length
 *   d: Head dimension
 *   Br: Row block size
 *   Bc: Column block size
 *
 * Output:
 *   O: Output matrix (N x d)
 *   L: logsumexp vector (N) - for backward pass
 */
void flash_attention_forward(
    const Matrix Q,    // N x d
    const Matrix K,    // N x d
    const Matrix V,    // N x d
    Matrix O,          // N x d (output)
    float *L,          // N (logsumexp output)
    int Br,            // Row block size
    int Bc             // Column block size
) {
    const int N = Q.rows;
    const int d = Q.cols;
    const float scale = 1.0f / sqrtf((float)d);

    const int Tr = (N + Br - 1) / Br;  // Number of row blocks
    const int Tc = (N + Bc - 1) / Bc;  // Number of column blocks

    // Process each row block
    for (int i = 0; i < Tr; i++) {
        const int row_start = i * Br;
        const int row_end = (row_start + Br < N) ? (row_start + Br) : N;
        const int rows_in_block = row_end - row_start;

        // Allocate temporary arrays for this row block
        float *m_i = (float*)malloc(rows_in_block * sizeof(float));  // Max values
        float *l_i = (float*)malloc(rows_in_block * sizeof(float));  // Sum of exponentials
        float *acc = (float*)calloc(rows_in_block * d, sizeof(float)); // Accumulated output

        // Initialize statistics
        for (int r = 0; r < rows_in_block; r++) {
            m_i[r] = -FLT_MAX;
            l_i[r] = 0.0f;
        }

        // Process each column block
        for (int j = 0; j < Tc; j++) {
            const int col_start = j * Bc;
            const int col_end = (col_start + Bc < N) ? (col_start + Bc) : N;
            const int cols_in_block = col_end - col_start;

            // Allocate temporary arrays for S and P matrices
            float *S = (float*)malloc(rows_in_block * cols_in_block * sizeof(float));
            float *P = (float*)malloc(rows_in_block * cols_in_block * sizeof(float));

            // Step 1: Compute S = Q_i @ K_j^T * scale
            for (int r = 0; r < rows_in_block; r++) {
                for (int c = 0; c < cols_in_block; c++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) {
                        sum += MAT(Q, row_start + r, k) * MAT(K, col_start + c, k);
                    }
                    S[r * cols_in_block + c] = sum * scale;
                }
            }

            // Step 2: Compute m_ij = rowmax(S) and update m_i
            float *m_ij = (float*)malloc(rows_in_block * sizeof(float));
            for (int r = 0; r < rows_in_block; r++) {
                float row_max = -FLT_MAX;
                for (int c = 0; c < cols_in_block; c++) {
                    float val = S[r * cols_in_block + c];
                    if (val > row_max) row_max = val;
                }
                m_ij[r] = max_float(m_i[r], row_max);
            }

            // Step 3: Compute P = exp(S - m_ij)
            for (int r = 0; r < rows_in_block; r++) {
                for (int c = 0; c < cols_in_block; c++) {
                    P[r * cols_in_block + c] = expf(S[r * cols_in_block + c] - m_ij[r]);
                }
            }

            // Step 4: Compute l_ij = rowsum(P)
            float *l_ij = (float*)malloc(rows_in_block * sizeof(float));
            for (int r = 0; r < rows_in_block; r++) {
                float sum = 0.0f;
                for (int c = 0; c < cols_in_block; c++) {
                    sum += P[r * cols_in_block + c];
                }
                l_ij[r] = sum;
            }

            // Step 5: Compute scaling factor alpha = exp(m_i - m_ij)
            float *alpha = (float*)malloc(rows_in_block * sizeof(float));
            for (int r = 0; r < rows_in_block; r++) {
                alpha[r] = expf(m_i[r] - m_ij[r]);
            }

            // Step 6: Update acc = acc * alpha + P @ V_j
            // First, scale acc by alpha
            for (int r = 0; r < rows_in_block; r++) {
                for (int k = 0; k < d; k++) {
                    acc[r * d + k] *= alpha[r];
                }
            }

            // Then, add P @ V_j
            for (int r = 0; r < rows_in_block; r++) {
                for (int k = 0; k < d; k++) {
                    float sum = 0.0f;
                    for (int c = 0; c < cols_in_block; c++) {
                        sum += P[r * cols_in_block + c] * MAT(V, col_start + c, k);
                    }
                    acc[r * d + k] += sum;
                }
            }

            // Step 7: Update l_i = l_i * alpha + l_ij
            for (int r = 0; r < rows_in_block; r++) {
                l_i[r] = l_i[r] * alpha[r] + l_ij[r];
            }

            // Step 8: Update m_i = m_ij
            for (int r = 0; r < rows_in_block; r++) {
                m_i[r] = m_ij[r];
            }

            // Free temporary arrays
            free(S);
            free(P);
            free(m_ij);
            free(l_ij);
            free(alpha);
        }

        // Step 9: Final normalization O_i = acc / l_i
        for (int r = 0; r < rows_in_block; r++) {
            for (int k = 0; k < d; k++) {
                MAT(O, row_start + r, k) = acc[r * d + k] / l_i[r];
            }
            // Store logsumexp for backward pass: L = m_i + log(l_i)
            L[row_start + r] = m_i[r] + logf(l_i[r]);
        }

        // Free row block arrays
        free(m_i);
        free(l_i);
        free(acc);
    }
}

// Reference implementation (standard attention) for verification
void standard_attention(
    const Matrix Q,
    const Matrix K,
    const Matrix V,
    Matrix O
) {
    const int N = Q.rows;
    const int d = Q.cols;
    const float scale = 1.0f / sqrtf((float)d);

    // Allocate S = Q @ K^T
    Matrix S = create_matrix(N, N);

    // Compute S = Q @ K^T * scale
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += MAT(Q, i, k) * MAT(K, j, k);
            }
            MAT(S, i, j) = sum * scale;
        }
    }

    // Apply softmax row-wise and compute O = softmax(S) @ V
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            if (MAT(S, i, j) > row_max) row_max = MAT(S, i, j);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            MAT(S, i, j) = expf(MAT(S, i, j) - row_max);
            sum += MAT(S, i, j);
        }

        // Normalize
        for (int j = 0; j < N; j++) {
            MAT(S, i, j) /= sum;
        }
    }

    // Compute O = S @ V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += MAT(S, i, j) * MAT(V, j, k);
            }
            MAT(O, i, k) = sum;
        }
    }

    free_matrix(&S);
}

// Test function
int main() {
    printf("FlashAttention-2 Forward Pass Test\n");
    printf("===================================\n\n");

    // Parameters
    const int N = 128;   // Sequence length
    const int d = 64;    // Head dimension
    const int Br = 32;   // Row block size
    const int Bc = 32;   // Column block size

    printf("Configuration:\n");
    printf("  Sequence length (N): %d\n", N);
    printf("  Head dimension (d):  %d\n", d);
    printf("  Row block size (Br): %d\n", Br);
    printf("  Col block size (Bc): %d\n\n", Bc);

    // Create matrices
    Matrix Q = create_matrix(N, d);
    Matrix K = create_matrix(N, d);
    Matrix V = create_matrix(N, d);
    Matrix O_flash = create_matrix(N, d);
    Matrix O_ref = create_matrix(N, d);
    float *L = (float*)malloc(N * sizeof(float));

    // Initialize with random values
    srand(42);
    init_random(Q);
    init_random(K);
    init_random(V);

    printf("Running FlashAttention forward pass...\n");
    flash_attention_forward(Q, K, V, O_flash, L, Br, Bc);
    printf("Done.\n\n");

    printf("Running reference implementation...\n");
    standard_attention(Q, K, V, O_ref);
    printf("Done.\n\n");

    // Compute error
    printf("Verifying correctness...\n");
    float max_error = 0.0f;
    float sum_error = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float error = abs_float(O_flash.data[i] - O_ref.data[i]);
        if (error > max_error) max_error = error;
        sum_error += error;
    }
    float avg_error = sum_error / (N * d);

    printf("  Max error: %.6e\n", max_error);
    printf("  Avg error: %.6e\n\n", avg_error);

    if (max_error < 1e-4f) {
        printf("[PASS] Test PASSED!\n");
    } else {
        printf("[FAIL] Test FAILED!\n");
    }

    // Cleanup
    free_matrix(&Q);
    free_matrix(&K);
    free_matrix(&V);
    free_matrix(&O_flash);
    free_matrix(&O_ref);
    free(L);

    return 0;
}
