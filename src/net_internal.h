#ifndef NET_INTERNAL_H
#define NET_INTERNAL_H

#include "internals.h"
#include "net.h"

double nn_estimated_flops_per_sample(Network *nn);
double compute_network_flops_per_sample(Network *nn);
double compute_loss_from_output(Network *nn, Matrix *output, Matrix *target,
                                int batch_size);
void copy_matrix_to_column_major(const Matrix *src, nn_float *dst);
void copy_column_major_to_matrix(Matrix *dst, const nn_float *src);
TrainStepStats nn_train_batch_cpu(Network *nn, Matrix *input, Matrix *target,
                                  int batch_size, Matrix *output_buffer);

#ifdef USE_CUDA
typedef struct GpuWorkspace GpuWorkspace;

ForwardCache *nn_forward_cached_gpu(Network *nn, Matrix *input);
Matrix *nn_forward_gpu(Network *nn, Matrix *input);
TrainStepStats nn_train_batch_gpu(Network *nn, Matrix *input, Matrix *target,
                                  int batch_size, Matrix *output_buffer);

GpuWorkspace *gpu_workspace_create(void);
void gpu_workspace_destroy(GpuWorkspace *workspace);
void gpu_workspace_set_seed(GpuWorkspace *workspace, unsigned long long seed);
void gpu_workspace_set_theoretical_tflops(GpuWorkspace *workspace, double tflops);
#endif

#endif /* NET_INTERNAL_H */
