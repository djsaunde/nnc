#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#ifdef USE_CUDA
#include <stddef.h>

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DenseLayerGpuContext DenseLayerGpuContext;

DenseLayerGpuContext *dense_gpu_create(int input_size, int output_size);
void dense_gpu_destroy(DenseLayerGpuContext *ctx);
void dense_gpu_set_weights(DenseLayerGpuContext *ctx, const nn_float *weights, const nn_float *bias);
void dense_gpu_forward(DenseLayerGpuContext *ctx, const nn_float *input, int batch, nn_float *output,
                       nn_float *last_input);
void dense_gpu_backward(DenseLayerGpuContext *ctx, const nn_float *grad_output,
                        const nn_float *last_input, int batch, nn_float *grad_weights,
                        nn_float *grad_input);
void dense_gpu_update(DenseLayerGpuContext *ctx, const nn_float *weights, const nn_float *bias);

#ifdef __cplusplus
}
#endif
#endif

#endif
