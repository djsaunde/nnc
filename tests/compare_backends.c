#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "net.h"
#include <cuda_runtime.h>

#ifndef USE_CUDA
int main(void)
{
    printf("USE_CUDA not enabled; skipping GPU comparison test.\n");
    return 0;
}
#else
static Matrix *create_random_matrix(int rows, int cols)
{
    Matrix *m = matrix_create(rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            m->data[r * cols + c] = (nn_float) (((double) rand() / RAND_MAX) * 2.0 - 1.0);
        }
    }
    return m;
}

static nn_float max_abs_diff(Matrix *a, Matrix *b)
{
    nn_float max_diff = 0.0f;
    int total = a->rows * a->cols;
    for (int i = 0; i < total; i++) {
        nn_float diff = fabsf(a->data[i] - b->data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main(void)
{
    const int input_size = 32;
    const int hidden_size = 16;
    const int output_size = 10;
    const int batch = 8;
    const nn_float learning_rate = 0.05f;
    const nn_float tol = 1e-4f;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        printf("No CUDA device available; skipping GPU comparison test.\n");
        return 0;
    }

    srand(1234);
    Network *cpu = nn_create_mlp(input_size, hidden_size, output_size, ACT_RELU, ACT_SOFTMAX,
                                  0.0, BACKEND_CPU);

    srand(1234);
    Network *gpu = nn_create_mlp(input_size, hidden_size, output_size, ACT_RELU, ACT_SOFTMAX,
                                  0.0, BACKEND_GPU);

    if (gpu == NULL) {
        fprintf(stderr, "Failed to create GPU backend network.\n");
        return 1;
    }

    Matrix *input = create_random_matrix(input_size, batch);
    Matrix *target = create_random_matrix(output_size, batch);

    nn_zero_gradients(cpu);
    nn_zero_gradients(gpu);

    TrainStepStats stats_cpu = nn_train_batch(cpu, input, target, batch, NULL);
    TrainStepStats stats_gpu = nn_train_batch(gpu, input, target, batch, NULL);

    if (fabsf(stats_cpu.loss - stats_gpu.loss) > tol) {
        fprintf(stderr, "Loss mismatch: CPU %.12f vs GPU %.12f\n", stats_cpu.loss,
                stats_gpu.loss);
        return 1;
    }
    if (fabsf(stats_cpu.grad_norm - stats_gpu.grad_norm) > tol) {
        fprintf(stderr, "Grad norm mismatch: CPU %.12f vs GPU %.12f\n",
                stats_cpu.grad_norm, stats_gpu.grad_norm);
        return 1;
    }

    nn_apply_gradients(cpu, learning_rate, batch);
    nn_apply_gradients(gpu, learning_rate, batch);

    Matrix *test_input = create_random_matrix(input_size, batch);
    Matrix *cpu_out = nn_forward(cpu, test_input);
    Matrix *gpu_out = nn_forward(gpu, test_input);

    nn_float diff = max_abs_diff(cpu_out, gpu_out);
    if (diff > tol) {
        fprintf(stderr, "Output mismatch after update: max diff %.12f\n", diff);
        return 1;
    }

    matrix_free(cpu_out);
    matrix_free(gpu_out);
    matrix_free(test_input);
    matrix_free(input);
    matrix_free(target);

    nn_free(cpu);
    nn_free(gpu);

    printf("CPU and GPU backends match within tolerance %.1e.\n", tol);
    return 0;
}
#endif
