// gemm.cu
// Naive CUDA GEMM (global memory only) + optional transpose + in-place update
// Build: nvcc -O2 gemm.cu -o gemm
// Run:   ./gemm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(1);                                                   \
    }                                                              \
} while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -------------------------
// Part 1: Naive GEMM kernel that writes D
// D = alpha*(A*B) + beta*C
// A: m x k, B: k x n, C: m x n, D: m x n
// -------------------------
__global__ void gemm_naive_out(
    int m, int n, int k,
    float alpha, const float* A, const float* B,
    float beta,  const float* C, float* D
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0,m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0,n)
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; t++) {
            sum += A[row * k + t] * B[t * n + col];
        }
        float c = C[row * n + col];
        D[row * n + col] = alpha * sum + beta * c;
    }
}

// -------------------------
// Part 2: Extended GEMM kernel (in-place update C)
// C <- alpha * op(A) * op(B) + beta * C
//
// transA = 0: op(A) = A, A is m x k
// transA = 1: op(A) = A^T, but A stored as k x m (so A^T is m x k)
//
// transB = 0: op(B) = B, B is k x n
// transB = 1: op(B) = B^T, but B stored as n x k (so B^T is k x n)
//
// NOTE: This convention matches typical GEMM APIs: when you request transpose,
// the *stored* matrix has swapped leading dimensions.
// -------------------------
__device__ __forceinline__ float loadA(
    const float* A, int m, int k, int row, int t, int transA
) {
    // Want op(A)[row, t] where op(A) is (m x k)
    // If transA==0: A is (m x k), A[row,t] = A[row*k + t]
    // If transA==1: A is stored as (k x m), and op(A)=A^T => op(A)[row,t] = A[t*m + row]
    return transA ? A[t * m + row] : A[row * k + t];
}

__device__ __forceinline__ float loadB(
    const float* B, int k, int n, int t, int col, int transB
) {
    // Want op(B)[t, col] where op(B) is (k x n)
    // If transB==0: B is (k x n), B[t,col] = B[t*n + col]
    // If transB==1: B is stored as (n x k), and op(B)=B^T => op(B)[t,col] = B[col*k + t]
    return transB ? B[col * k + t] : B[t * n + col];
}

__global__ void gemm_naive_inplace_trans(
    int m, int n, int k,
    float alpha, const float* A, const float* B,
    float beta, float* C,
    int transA, int transB
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0,m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0,n)
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; t++) {
            sum += loadA(A, m, k, row, t, transA) * loadB(B, k, n, t, col, transB);
        }
        float oldc = C[row * n + col];
        C[row * n + col] = alpha * sum + beta * oldc;
    }
}

// -------------------------
// CPU reference (same semantics as extended kernel)
// C <- alpha * op(A) * op(B) + beta * C
// A stored as (m x k) if transA=0 else (k x m)
// B stored as (k x n) if transB=0 else (n x k)
// -------------------------
static inline float cpu_loadA(const std::vector<float>& A, int m, int k, int row, int t, int transA) {
    return transA ? A[t * m + row] : A[row * k + t];
}
static inline float cpu_loadB(const std::vector<float>& B, int k, int n, int t, int col, int transB) {
    return transB ? B[col * k + t] : B[t * n + col];
}

void cpu_gemm_inplace(
    int m, int n, int k,
    float alpha, const std::vector<float>& A, const std::vector<float>& B,
    float beta, std::vector<float>& C,
    int transA, int transB
) {
    std::vector<float> out(C.size(), 0.0f);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++) {
                sum += cpu_loadA(A, m, k, i, t, transA) * cpu_loadB(B, k, n, t, j, transB);
            }
            out[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
    C.swap(out);
}

// -------------------------
// Helpers: random init + compare
// -------------------------
void fill_rand(std::vector<float>& x, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
}

bool allclose(const std::vector<float>& a, const std::vector<float>& b, float atol=1e-4f, float rtol=1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::max(std::fabs(a[i]), std::fabs(b[i]));
        if (diff > tol) return false;
    }
    return true;
}

// Build stored A/B with transpose convention:
// - if transA=0: store A as (m x k)
// - if transA=1: store A as (k x m) i.e. physical A = transpose(logical m x k)
// same for B
std::vector<float> make_A_storage(int m, int k, int transA, const std::vector<float>& A_mk) {
    if (!transA) return A_mk; // m x k
    std::vector<float> A_km((size_t)k * (size_t)m);
    for (int i = 0; i < m; i++)
        for (int t = 0; t < k; t++)
            A_km[t * m + i] = A_mk[i * k + t]; // transpose
    return A_km;
}
std::vector<float> make_B_storage(int k, int n, int transB, const std::vector<float>& B_kn) {
    if (!transB) return B_kn; // k x n
    std::vector<float> B_nk((size_t)n * (size_t)k);
    for (int t = 0; t < k; t++)
        for (int j = 0; j < n; j++)
            B_nk[j * k + t] = B_kn[t * n + j]; // transpose
    return B_nk;
}

// -------------------------
// One test run for extended kernel
// -------------------------
void run_one_test(int m, int n, int k, int transA, int transB, std::mt19937& rng) {
    // logical A(m x k), logical B(k x n)
    std::vector<float> A_mk((size_t)m * (size_t)k);
    std::vector<float> B_kn((size_t)k * (size_t)n);
    std::vector<float> C_mn((size_t)m * (size_t)n);

    fill_rand(A_mk, rng);
    fill_rand(B_kn, rng);
    fill_rand(C_mn, rng);

    // stored representations according to transpose flags
    std::vector<float> A = make_A_storage(m, k, transA, A_mk);
    std::vector<float> B = make_B_storage(k, n, transB, B_kn);

    // CPU reference
    std::vector<float> C_ref = C_mn;
    float alpha = 1.25f, beta = -0.75f;
    cpu_gemm_inplace(m, n, k, alpha, A, B, beta, C_ref, transA, transB);

    // GPU
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, C_mn.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C_mn.data(), C_mn.size()*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(ceil_div(n, block.x), ceil_div(m, block.y));
    gemm_naive_inplace_trans<<<grid, block>>>(m, n, k, alpha, dA, dB, beta, dC, transA, transB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_gpu(C_mn.size());
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    if (!allclose(C_ref, C_gpu)) {
        fprintf(stderr, "FAIL m=%d n=%d k=%d transA=%d transB=%d\n", m, n, k, transA, transB);
        for (int i = 0; i < std::min(5, m*n); i++) {
            fprintf(stderr, "  idx %d: ref=%f gpu=%f\n", i, C_ref[i], C_gpu[i]);
        }
        exit(2);
    }
}

// -------------------------
// Test naive-out kernel quickly by comparing to CPU (no transpose, writes D)
// -------------------------
void run_one_test_naive_out(int m, int n, int k, std::mt19937& rng) {
    std::vector<float> A((size_t)m*(size_t)k);
    std::vector<float> B((size_t)k*(size_t)n);
    std::vector<float> C((size_t)m*(size_t)n);
    fill_rand(A, rng); fill_rand(B, rng); fill_rand(C, rng);

    float alpha=0.9f, beta=1.1f;

    // CPU compute D
    std::vector<float> D_ref((size_t)m*(size_t)n, 0.0f);
    for (int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum=0.0f;
            for(int t=0;t<k;t++){
                sum += A[i*k+t]*B[t*n+j];
            }
            D_ref[i*n+j] = alpha*sum + beta*C[i*n+j];
        }
    }

    // GPU compute D
    float *dA=nullptr,*dB=nullptr,*dC=nullptr,*dD=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, C.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, D_ref.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C.data(), C.size()*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid(ceil_div(n, block.x), ceil_div(m, block.y));
    gemm_naive_out<<<grid, block>>>(m,n,k,alpha,dA,dB,beta,dC,dD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> D_gpu(D_ref.size());
    CUDA_CHECK(cudaMemcpy(D_gpu.data(), dD, D_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dD));

    if (!allclose(D_ref, D_gpu)) {
        fprintf(stderr, "FAIL naive-out m=%d n=%d k=%d\n", m,n,k);
        exit(3);
    }
}

int main() {
    // Print device info
    int dev=0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Using GPU: %s\n", prop.name);

    std::mt19937 rng(20260206);

    // ---- Required corner cases + additional tests ----
    // (m,n,k) correspond to op(A) (m x k) and op(B) (k x n)
    struct Case { int m,n,k; } cases[] = {
        {1,1,1},
        {1,5,1},
        {2,3,1},
        {2,2,2},
        {3,4,5},
        {17,19,23}, // non-multiple of block
        {64,64,64},
        {128,96,80}
    };

    // Test naive-out kernel (no transpose, writes D)
    for (auto &cs : cases) {
        run_one_test_naive_out(cs.m, cs.n, cs.k, rng);
    }
    printf("[PASS] naive GEMM (writes D) tests passed.\n");

    // Extended kernel tests: transA/transB combinations
    for (auto &cs : cases) {
        for (int transA = 0; transA <= 1; transA++) {
            for (int transB = 0; transB <= 1; transB++) {
                run_one_test(cs.m, cs.n, cs.k, transA, transB, rng);
            }
        }
    }

    // More randomized tests
    for (int t = 0; t < 50; t++) {
        int m = 1 + (rng() % 64);
        int n = 1 + (rng() % 64);
        int k = 1 + (rng() % 64);
        int transA = rng() % 2;
        int transB = rng() % 2;
        run_one_test(m, n, k, transA, transB, rng);
    }

    printf("[PASS] extended GEMM (transpose + in-place C) tests passed ✅\n");
    printf("All done.\n");
    return 0;
}