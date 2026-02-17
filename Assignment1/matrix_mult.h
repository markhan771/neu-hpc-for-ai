#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

// Matrix structure
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

// Thread data structure for multi-threaded implementation
typedef struct {
    Matrix *A;
    Matrix *B;
    Matrix *C;
    int start_row;
    int end_row;
} ThreadData;

// Function prototypes
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *mat);
void fill_matrix_random(Matrix *mat);
void fill_matrix_value(Matrix *mat, double value);
void print_matrix(Matrix *mat);
int compare_matrices(Matrix *A, Matrix *B, double tolerance);

// Single-threaded matrix multiplication
Matrix* matrix_mult_single(Matrix *A, Matrix *B);

// Multi-threaded matrix multiplication
Matrix* matrix_mult_multi(Matrix *A, Matrix *B, int num_threads);

#endif // MATRIX_MULT_H
