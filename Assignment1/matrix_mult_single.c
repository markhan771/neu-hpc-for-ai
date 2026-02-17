#include "matrix_mult.h"

// Create a matrix with given dimensions
Matrix* create_matrix(int rows, int cols) {
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Error: Memory allocation failed for matrix structure\n");
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;

    // Allocate array of row pointers
    mat->data = (double**)malloc(rows * sizeof(double*));
    if (!mat->data) {
        fprintf(stderr, "Error: Memory allocation failed for matrix rows\n");
        free(mat);
        return NULL;
    }

    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double*)calloc(cols, sizeof(double));
        if (!mat->data[i]) {
            fprintf(stderr, "Error: Memory allocation failed for matrix row %d\n", i);
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(mat->data[j]);
            }
            free(mat->data);
            free(mat);
            return NULL;
        }
    }

    return mat;
}

// Free matrix memory
void free_matrix(Matrix *mat) {
    if (mat) {
        if (mat->data) {
            for (int i = 0; i < mat->rows; i++) {
                free(mat->data[i]);
            }
            free(mat->data);
        }
        free(mat);
    }
}

// Fill matrix with random values between 0 and 1
void fill_matrix_random(Matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// Fill matrix with a specific value
void fill_matrix_value(Matrix *mat, double value) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = value;
        }
    }
}

// Print matrix
void print_matrix(Matrix *mat) {
    printf("Matrix (%dx%d):\n", mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%8.4f ", mat->data[i][j]);
        }
        printf("\n");
    }
}

// Compare two matrices within a tolerance
int compare_matrices(Matrix *A, Matrix *B, double tolerance) {
    if (A->rows != B->rows || A->cols != B->cols) {
        return 0;
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            double diff = A->data[i][j] - B->data[i][j];
            if (diff < 0) diff = -diff;
            if (diff > tolerance) {
                return 0;
            }
        }
    }

    return 1;
}

// Single-threaded matrix multiplication
// C = A * B
// A is (m x n), B is (n x p), C will be (m x p)
Matrix* matrix_mult_single(Matrix *A, Matrix *B) {
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

    // Perform multiplication
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }

    return C;
}
