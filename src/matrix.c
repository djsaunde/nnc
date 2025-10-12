#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix *matrix_create(int rows, int cols)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (nn_float *) malloc((size_t) rows * cols * sizeof(nn_float));
    return m;
}

void matrix_free(Matrix *m)
{
    free(m->data);
    free(m);
}

nn_float matrix_get(Matrix *m, int i, int j)
{
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, int i, int j, nn_float value)
{
    m->data[i * m->cols + j] = value;
}

void matrix_print(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%0.4f ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}

void matrix_randomize(Matrix *m)
{
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (nn_float) (((double) rand() / RAND_MAX) * 2.0 - 1.0);
    }
}

void matrix_fill(Matrix *m, nn_float value)
{
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = value;
    }
}

Matrix *matrix_multiply(Matrix *a, Matrix *b)
{
    if (a->cols != b->rows) {
        printf("Error: Matrix dims don't match for multiplication\n");
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            nn_float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(result, i, j, sum);
        }
    }

    return result;
}

Matrix *matrix_add(Matrix *a, Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Matrix dims don't match for addition\n");
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix *matrix_subtract(Matrix *a, Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Matrix dims don't match for subtraction\n");
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

Matrix *matrix_multiply_elementwise(Matrix *a, Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: Matrix dims don't match for element-wise multiplication\n");
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

Matrix *matrix_transpose(Matrix *m)
{
    Matrix *result = matrix_create(m->cols, m->rows);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix_set(result, j, i, matrix_get(m, i, j));
        }
    }
    return result;
}

void matrix_apply(Matrix *m, nn_float (*func)(nn_float))
{
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = func(m->data[i]);
    }
}

// Multiply all elements by a scalar
void matrix_scale(Matrix *m, nn_float scalar)
{
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] *= scalar;
    }
}

// Create a copy of a matrix
Matrix *matrix_copy(Matrix *m)
{
    Matrix *copy = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        copy->data[i] = m->data[i];
    }
    return copy;
}

void matrix_add_inplace(Matrix *dest, Matrix *src)
{
    if (dest->rows != src->rows || dest->cols != src->cols) {
        printf("Error: Matrix dims don't match for inplace addition\n");
        return;
    }

    int total = dest->rows * dest->cols;
    for (int i = 0; i < total; i++) {
        dest->data[i] += src->data[i];
    }
}

void matrix_copy_into(Matrix *dest, Matrix *src)
{
    if (dest->rows != src->rows || dest->cols != src->cols) {
        printf("Error: Matrix dims don't match for copy\n");
        return;
    }

    memcpy(dest->data, src->data, (size_t) dest->rows * dest->cols * sizeof(nn_float));
}

void matrix_zero(Matrix *m)
{
    memset(m->data, 0, (size_t) m->rows * m->cols * sizeof(nn_float));
}
