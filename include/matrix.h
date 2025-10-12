#ifndef MATRIX_H
#define MATRIX_H

#include "nn_types.h"

typedef struct {
    int rows;
    int cols;
    nn_float *data;
} Matrix;

// Basic ops
Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
nn_float matrix_get(Matrix *m, int i, int j);
void matrix_set(Matrix *m, int i, int j, nn_float value);
void matrix_print(Matrix *m);

// Initialization
void matrix_randomize(Matrix *m);
void matrix_fill(Matrix *m, nn_float value);

// Matrix operations
Matrix *matrix_multiply(Matrix *a, Matrix *b);
Matrix *matrix_add(Matrix *a, Matrix *b);
Matrix *matrix_subtract(Matrix *a, Matrix *b);
Matrix *matrix_multiply_elementwise(Matrix *a, Matrix *b);
void matrix_apply(Matrix *m, nn_float (*func)(nn_float));
Matrix *matrix_transpose(Matrix *m);
void matrix_scale(Matrix *m, nn_float scalar);
Matrix *matrix_copy(Matrix *m);
void matrix_add_inplace(Matrix *dest, Matrix *src);
void matrix_copy_into(Matrix *dest, Matrix *src);
void matrix_zero(Matrix *m);

#endif
