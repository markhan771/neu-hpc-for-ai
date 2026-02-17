#include "matrix_mult.h"

// Thread function for matrix multiplication
// Each thread computes a range of rows in the result matrix
void* thread_matrix_mult(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    Matrix *A = data->A;
    Matrix *B = data->B;
    Matrix *C = data->C;

    // Compute assigned rows
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }

    pthread_exit(NULL);
}

// Multi-threaded matrix multiplication
// C = A * B using num_threads threads
Matrix* matrix_mult_multi(Matrix *A, Matrix *B, int num_threads) {
    // Check if multiplication is valid
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: Matrix dimensions incompatible for multiplication\n");
        fprintf(stderr, "A is %dx%d, B is %dx%d\n", A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }

    // Create result matrix
    Matrix *C = create_matrix(A->rows, B->cols);
    if (!C) {
        return NULL;
    }

    // If num_threads is greater than number of rows, limit it
    if (num_threads > A->rows) {
        num_threads = A->rows;
    }

    // Create threads and thread data
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

    if (!threads || !thread_data) {
        fprintf(stderr, "Error: Memory allocation failed for threads\n");
        free_matrix(C);
        free(threads);
        free(thread_data);
        return NULL;
    }

    // Divide work among threads
    int rows_per_thread = A->rows / num_threads;
    int remaining_rows = A->rows % num_threads;

    int current_row = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].start_row = current_row;

        // Distribute remaining rows to first few threads
        int rows_for_this_thread = rows_per_thread + (i < remaining_rows ? 1 : 0);
        thread_data[i].end_row = current_row + rows_for_this_thread;

        current_row = thread_data[i].end_row;

        // Create thread
        if (pthread_create(&threads[i], NULL, thread_matrix_mult, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: Failed to create thread %d\n", i);
            // Clean up
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free_matrix(C);
            free(threads);
            free(thread_data);
            return NULL;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    free(threads);
    free(thread_data);

    return C;
}
