#ifndef INTERNALS_H
#define INTERNALS_H

#include "net.h"
#include "matrix.h"

struct DenseLayerGpuContext;
struct Conv2DLayerGpuContext;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SkipConnectionData {
    Matrix *saved_input;
    Matrix *grad_accum;
    int refcount;
    Layer *projection_conv;
    Layer *projection_bn;
#ifdef USE_CUDA
    nn_float *d_saved_input;
    nn_float *d_grad_accum;
    size_t saved_capacity;
    size_t grad_capacity;
#endif
} SkipConnectionData;

typedef struct DenseLayerData {
    Matrix *weights;
    Matrix *bias;
    Matrix *last_input;
    Matrix *grad_weights;
    Matrix *grad_bias;
    Matrix *output_cache;
#ifdef USE_CUDA
    DenseLayerGpuContext *gpu;
#endif
    Matrix *adam_m_weights;
    Matrix *adam_v_weights;
    Matrix *adam_m_bias;
    Matrix *adam_v_bias;
    int adam_initialized;
#ifdef USE_CUDA
    int adam_device_initialized;
#endif
    BackendKind backend;
    int weights_dirty;
} DenseLayerData;

typedef struct ActivationLayerData {
    ActivationFn func;
    ActivationDerivativeFn derivative;
    ActivationKind kind;
    Matrix *last_input;
    Matrix *last_output;
#ifdef USE_CUDA
    nn_float *d_last_output;
    int last_rows;
    int last_cols;
#endif
} ActivationLayerData;

typedef struct DropoutLayerData {
    nn_float rate;
    nn_float keep_probability;
    int training;
    Matrix *mask;
#ifdef USE_CUDA
    nn_float *d_mask;
    size_t mask_capacity;
#endif
} DropoutLayerData;

typedef struct SoftmaxLayerData {
    Matrix *last_input;
    Matrix *last_output;
#ifdef USE_CUDA
    nn_float *d_last_output;
    int last_rows;
    int last_cols;
#endif
} SoftmaxLayerData;

typedef struct BatchNormLayerData {
    int channels;
    int spatial;
    nn_float epsilon;
    nn_float momentum;
    BackendKind backend;
    Matrix *gamma;
    Matrix *beta;
    Matrix *running_mean;
    Matrix *running_var;
    Matrix *batch_mean;
    Matrix *batch_var;
    Matrix *normalized;
    Matrix *input_cache;
    Matrix *grad_gamma;
    Matrix *grad_beta;
    Matrix *adam_m_gamma;
    Matrix *adam_v_gamma;
    Matrix *adam_m_beta;
    Matrix *adam_v_beta;
    int adam_initialized;
    int training;
#ifdef USE_CUDA
    nn_float *d_gamma;
    nn_float *d_beta;
    nn_float *d_running_mean;
    nn_float *d_running_var;
    nn_float *d_batch_mean;
    nn_float *d_batch_var;
    nn_float *d_normalized;
    nn_float *d_input_cache;
    nn_float *d_grad_input;
    nn_float *d_grad_gamma;
    nn_float *d_grad_beta;
    nn_float *d_m_gamma;
    nn_float *d_v_gamma;
    nn_float *d_m_beta;
    nn_float *d_v_beta;
    int adam_device_initialized;
#endif
    int gpu_capacity;
} BatchNormLayerData;

typedef struct MaxPoolLayerData {
    int channels;
    int input_height;
    int input_width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int output_height;
    int output_width;
    Matrix *output_cache;
    int *max_indices;
#ifdef USE_CUDA
    int *d_max_indices;
#endif
} MaxPoolLayerData;

typedef struct GlobalAvgPoolLayerData {
    int channels;
    int input_height;
    int input_width;
    Matrix *output_cache;
#ifdef USE_CUDA
    nn_float *d_output_cache;
#endif
} GlobalAvgPoolLayerData;

typedef struct Conv2DLayerData {
    int in_channels;
    int out_channels;
    int input_height;
    int input_width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int output_height;
    int output_width;
    Matrix *weights;
    Matrix *bias;
    Matrix *last_input;
    Matrix *output_cache;
    Matrix *grad_weights;
    Matrix *grad_bias;
    Matrix *adam_m_weights;
    Matrix *adam_v_weights;
    Matrix *adam_m_bias;
    Matrix *adam_v_bias;
    int adam_initialized;
#ifdef USE_CUDA
    int adam_device_initialized;
    Conv2DLayerGpuContext *gpu;
#endif
    BackendKind backend;
} Conv2DLayerData;

Matrix *ensure_matrix(Matrix *matrix, int rows, int cols);

void dense_layer_apply_sgd(DenseLayerData *data, nn_float learning_rate, nn_float weight_decay);
void dense_layer_apply_adamw(DenseLayerData *data, const OptimizerState *opt,
                             nn_float learning_rate, nn_float inv_bias_correction1,
                             nn_float inv_bias_correction2, nn_float grad_scale);

void conv2d_layer_apply_sgd(Conv2DLayerData *data, nn_float learning_rate, nn_float weight_decay);
void conv2d_layer_apply_adamw(Conv2DLayerData *data, const OptimizerState *opt,
                              nn_float learning_rate, nn_float inv_bias_correction1,
                              nn_float inv_bias_correction2, nn_float grad_scale);

#ifdef USE_CUDA
void batchnorm_layer_gpu_ensure_capacity(BatchNormLayerData *data, int batch);
void batchnorm_layer_gpu_copy_params_to_device(BatchNormLayerData *data);
void batchnorm_layer_gpu_copy_params_to_host(BatchNormLayerData *data);
void batchnorm_layer_gpu_copy_running_to_device(BatchNormLayerData *data);
void batchnorm_layer_gpu_copy_running_to_host(BatchNormLayerData *data);
void batchnorm_layer_gpu_copy_grads_to_host(BatchNormLayerData *data);
#endif

void batchnorm_layer_apply_sgd(BatchNormLayerData *data, nn_float learning_rate,
                               nn_float weight_decay, nn_float grad_scale);
void batchnorm_layer_apply_adamw(BatchNormLayerData *data, const OptimizerState *opt,
                                 nn_float learning_rate, nn_float inv_bias_correction1,
                                 nn_float inv_bias_correction2, nn_float grad_scale);
void batchnorm_layer_zero_gradients(BatchNormLayerData *data);

Layer *layer_dense_create_backend(BackendKind backend, int input_size, int output_size);
Layer *layer_activation_from_kind_backend(BackendKind backend, ActivationKind kind);
Layer *layer_conv2d_create_backend(BackendKind backend, int in_channels, int out_channels,
                                   int input_height, int input_width, int kernel_h, int kernel_w,
                                   int stride_h, int stride_w, int pad_h, int pad_w);

#ifdef __cplusplus
}
#endif

#endif
