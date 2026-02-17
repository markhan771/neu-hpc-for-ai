#include "matrix_mult.h"
#include <sys/time.h>

// Get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Run benchmark for a specific configuration
void run_benchmark(int size, int num_threads) {
    printf("Testing %dx%d matrices with %d thread(s)...\n", size, size, num_threads);

    // Create matrices
    Matrix *A = create_matrix(size, size);
    Matrix *B = create_matrix(size, size);

    if (!A || !B) {
        fprintf(stderr, "Failed to allocate matrices\n");
        free_matrix(A);
        free_matrix(B);
        return;
    }

    // Fill with random values
    fill_matrix_random(A);
    fill_matrix_random(B);

    // Warm-up run
    Matrix *C_warmup = matrix_mult_multi(A, B, num_threads);
    if (!C_warmup) {
        fprintf(stderr, "Warm-up run failed\n");
        free_matrix(A);
        free_matrix(B);
        return;
    }
    free_matrix(C_warmup);

    // Timed run
    double start_time = get_time();
    Matrix *C = matrix_mult_multi(A, B, num_threads);
    double end_time = get_time();

    if (!C) {
        fprintf(stderr, "Benchmark run failed\n");
        free_matrix(A);
        free_matrix(B);
        return;
    }

    double elapsed_time = end_time - start_time;
    double gflops = (2.0 * size * size * size) / (elapsed_time * 1e9);

    printf("  Time: %.4f seconds\n", elapsed_time);
    printf("  Performance: %.2f GFLOPS\n", gflops);

    // Clean up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
}

// Run comprehensive benchmark
void run_comprehensive_benchmark(int size) {
    printf("\n==============================================\n");
    printf("Benchmark: %dx%d Matrix Multiplication\n", size, size);
    printf("==============================================\n");

    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_configs = 6;

    double baseline_time = 0.0;
    double times[6];

    for (int i = 0; i < num_configs; i++) {
        int num_threads = thread_counts[i];

        // Create matrices
        Matrix *A = create_matrix(size, size);
        Matrix *B = create_matrix(size, size);

        if (!A || !B) {
            fprintf(stderr, "Failed to allocate matrices\n");
            free_matrix(A);
            free_matrix(B);
            continue;
        }

        fill_matrix_random(A);
        fill_matrix_random(B);

        // Warm-up
        Matrix *C_warmup = matrix_mult_multi(A, B, num_threads);
        if (C_warmup) {
            free_matrix(C_warmup);
        }

        // Benchmark run
        double start_time = get_time();
        Matrix *C = matrix_mult_multi(A, B, num_threads);
        double end_time = get_time();

        if (C) {
            double elapsed_time = end_time - start_time;
            times[i] = elapsed_time;

            if (i == 0) {
                baseline_time = elapsed_time;
            }

            double speedup = baseline_time / elapsed_time;
            double gflops = (2.0 * size * size * size) / (elapsed_time * 1e9);

            printf("\nThreads: %3d\n", num_threads);
            printf("  Time:     %10.4f seconds\n", elapsed_time);
            printf("  Speedup:  %10.2fx\n", speedup);
            printf("  GFLOPS:   %10.2f\n", gflops);

            free_matrix(C);
        } else {
            fprintf(stderr, "Multiplication failed for %d threads\n", num_threads);
            times[i] = -1.0;
        }

        free_matrix(A);
        free_matrix(B);
    }

    // Print summary table
    printf("\n==============================================\n");
    printf("Summary Table\n");
    printf("==============================================\n");
    printf("Threads | Time (s) | Speedup | Efficiency\n");
    printf("--------|----------|---------|------------\n");

    for (int i = 0; i < num_configs; i++) {
        if (times[i] > 0) {
            double speedup = baseline_time / times[i];
            double efficiency = speedup / thread_counts[i] * 100.0;
            printf("%7d | %8.4f | %7.2fx | %9.1f%%\n",
                   thread_counts[i], times[i], speedup, efficiency);
        } else {
            printf("%7d | FAILED\n", thread_counts[i]);
        }
    }

    printf("==============================================\n");
}

int main(int argc, char *argv[]) {
    printf("==============================================\n");
    printf("Matrix Multiplication Performance Benchmark\n");
    printf("==============================================\n");

    srand(time(NULL));

    // Default matrix size
    int size = 1000;

    // Parse command line argument if provided
    if (argc > 1) {
        size = atoi(argv[1]);
        if (size <= 0) {
            fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
            fprintf(stderr, "Usage: %s [matrix_size]\n", argv[0]);
            return 1;
        }
    }

    printf("Matrix size: %dx%d\n", size, size);
    printf("Total operations: %.2e\n", 2.0 * size * size * size);

    // Run comprehensive benchmark
    run_comprehensive_benchmark(size);

    // Test with different matrix sizes
    printf("\n\n==============================================\n");
    printf("Scaling Test: Different Matrix Sizes\n");
    printf("==============================================\n");

    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;
    int test_threads = 4;

    printf("\nUsing %d threads for scaling test\n\n", test_threads);
    printf("Size  | Time (s) | GFLOPS\n");
    printf("------|----------|--------\n");

    for (int i = 0; i < num_sizes; i++) {
        int test_size = sizes[i];

        Matrix *A = create_matrix(test_size, test_size);
        Matrix *B = create_matrix(test_size, test_size);

        if (!A || !B) {
            printf("%5d | FAILED (allocation)\n", test_size);
            free_matrix(A);
            free_matrix(B);
            continue;
        }

        fill_matrix_random(A);
        fill_matrix_random(B);

        double start_time = get_time();
        Matrix *C = matrix_mult_multi(A, B, test_threads);
        double end_time = get_time();

        if (C) {
            double elapsed_time = end_time - start_time;
            double gflops = (2.0 * test_size * test_size * test_size) / (elapsed_time * 1e9);
            printf("%5d | %8.4f | %6.2f\n", test_size, elapsed_time, gflops);
            free_matrix(C);
        } else {
            printf("%5d | FAILED (computation)\n", test_size);
        }

        free_matrix(A);
        free_matrix(B);
    }

    printf("==============================================\n");

    return 0;
}
