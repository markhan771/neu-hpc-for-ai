#include "matrix_mult.h"

// Test case structure
typedef struct {
    int A_rows;
    int A_cols;
    int B_rows;
    int B_cols;
    const char *description;
} TestCase;

// Run a single test case
int run_test_case(TestCase tc, int test_num) {
    printf("\n--- Test %d: %s ---\n", test_num, tc.description);
    printf("A: %dx%d, B: %dx%d\n", tc.A_rows, tc.A_cols, tc.B_rows, tc.B_cols);

    // Create matrices
    Matrix *A = create_matrix(tc.A_rows, tc.A_cols);
    Matrix *B = create_matrix(tc.B_rows, tc.B_cols);

    if (!A || !B) {
        fprintf(stderr, "Failed to create matrices\n");
        free_matrix(A);
        free_matrix(B);
        return 0;
    }

    // Fill with random values
    fill_matrix_random(A);
    fill_matrix_random(B);

    // Print matrices if small enough
    if (tc.A_rows <= 3 && tc.A_cols <= 3 && tc.B_rows <= 3 && tc.B_cols <= 3) {
        print_matrix(A);
        print_matrix(B);
    }

    // Test single-threaded version
    Matrix *C_single = matrix_mult_single(A, B);
    if (!C_single) {
        fprintf(stderr, "Single-threaded multiplication failed\n");
        free_matrix(A);
        free_matrix(B);
        return 0;
    }

    printf("Single-threaded: Result is %dx%d\n", C_single->rows, C_single->cols);

    // Test multi-threaded version with different thread counts
    int thread_counts[] = {1, 2, 4, 8};
    int num_thread_tests = 4;

    for (int i = 0; i < num_thread_tests; i++) {
        int num_threads = thread_counts[i];
        Matrix *C_multi = matrix_mult_multi(A, B, num_threads);

        if (!C_multi) {
            fprintf(stderr, "Multi-threaded multiplication failed with %d threads\n", num_threads);
            free_matrix(A);
            free_matrix(B);
            free_matrix(C_single);
            return 0;
        }

        // Compare results
        if (!compare_matrices(C_single, C_multi, 1e-9)) {
            fprintf(stderr, "FAILED: Results differ between single and multi-threaded (%d threads)\n", num_threads);
            free_matrix(A);
            free_matrix(B);
            free_matrix(C_single);
            free_matrix(C_multi);
            return 0;
        }

        printf("Multi-threaded (%2d threads): PASSED\n", num_threads);
        free_matrix(C_multi);
    }

    // Print result if small enough
    if (C_single->rows <= 3 && C_single->cols <= 3) {
        print_matrix(C_single);
    }

    // Clean up
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_single);

    printf("Test %d: PASSED ✓\n", test_num);
    return 1;
}

int main() {
    printf("==============================================\n");
    printf("Matrix Multiplication Test Suite\n");
    printf("==============================================\n");

    srand(time(NULL));

    // Define test cases
    TestCase tests[] = {
        // Corner cases
        {1, 1, 1, 1, "1x1 * 1x1 (smallest)"},
        {1, 1, 1, 5, "1x1 * 1x5 (row vector result)"},
        {5, 1, 1, 1, "5x1 * 1x1 (column vector)"},
        {2, 1, 1, 3, "2x1 * 1x3"},
        {1, 3, 3, 1, "1x3 * 3x1 (vector dot product)"},

        // Small square matrices
        {2, 2, 2, 2, "2x2 * 2x2 (small square)"},
        {3, 3, 3, 3, "3x3 * 3x3"},
        {4, 4, 4, 4, "4x4 * 4x4"},

        // Rectangular matrices
        {2, 3, 3, 2, "2x3 * 3x2"},
        {3, 2, 2, 3, "3x2 * 2x3"},
        {5, 3, 3, 7, "5x3 * 3x7"},
        {10, 5, 5, 8, "10x5 * 5x8"},

        // Larger matrices
        {100, 100, 100, 100, "100x100 * 100x100"},
        {200, 150, 150, 100, "200x150 * 150x100"},
        {50, 200, 200, 50, "50x200 * 200x50"},

        // Edge cases with large aspect ratios
        {1, 100, 100, 1, "1x100 * 100x1"},
        {100, 1, 1, 100, "100x1 * 1x100"},
        {500, 2, 2, 500, "500x2 * 2x500"},
    };

    int num_tests = sizeof(tests) / sizeof(TestCase);
    int passed = 0;
    int failed = 0;

    // Run all tests
    for (int i = 0; i < num_tests; i++) {
        if (run_test_case(tests[i], i + 1)) {
            passed++;
        } else {
            failed++;
            printf("Test %d: FAILED ✗\n", i + 1);
        }
    }

    // Summary
    printf("\n==============================================\n");
    printf("Test Summary\n");
    printf("==============================================\n");
    printf("Total tests: %d\n", num_tests);
    printf("Passed:      %d\n", passed);
    printf("Failed:      %d\n", failed);
    printf("Success rate: %.1f%%\n", (100.0 * passed) / num_tests);
    printf("==============================================\n");

    return (failed == 0) ? 0 : 1;
}
