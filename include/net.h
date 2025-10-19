#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

#include "matrix.h"
#ifdef USE_CUDA
#include "gpu_backend.h"
typedef struct GpuWorkspace GpuWorkspace;
#endif

typedef nn_float (*ActivationFn)(nn_float value);
typedef nn_float (*ActivationDerivativeFn)(nn_float activated, nn_float pre_activation);

typedef enum {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_ACTIVATION,
    LAYER_DROPOUT,
    LAYER_SOFTMAX,
    LAYER_BATCHNORM,
    LAYER_MAXPOOL,
    LAYER_GLOBAL_AVG_POOL,
    LAYER_SKIP_SAVE,
    LAYER_SKIP_ADD
} LayerType;

typedef enum ActivationKind {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    ACT_SOFTMAX
} ActivationKind;

typedef enum { BACKEND_CPU, BACKEND_GPU } BackendKind;

typedef enum { OPTIMIZER_SGD, OPTIMIZER_ADAMW } OptimizerKind;

typedef struct {
    OptimizerKind kind;
    nn_float beta1;
    nn_float beta2;
    nn_float epsilon;
    nn_float weight_decay;
    unsigned long long step;
    nn_float beta1_power;
    nn_float beta2_power;
} OptimizerState;

typedef struct Layer {
    LayerType type;
    void *data;
    Matrix *(*forward)(struct Layer *layer, Matrix *input);
    Matrix *(*backward)(struct Layer *layer, Matrix *grad_output);
    void (*update)(struct Layer *layer, nn_float learning_rate);
    void (*destroy)(struct Layer *layer);
} Layer;

typedef struct {
    size_t layer_count;
    size_t capacity;
    Layer **layers;
    int input_size;
    int output_size;
    BackendKind backend;
    OptimizerState optimizer;
#ifdef USE_CUDA
    GpuWorkspace *gpu_workspace;
#endif
    double flops_per_sample;
    int flops_dirty;
    int log_memory;
} Network;

typedef struct {
    Matrix *output;
} ForwardCache;

typedef struct {
    Matrix *output_gradient;
    Matrix *input_gradient;
    nn_float loss;
    nn_float grad_norm;
} BackwardCache;

typedef struct {
    nn_float loss;
    nn_float grad_norm;
    int samples;
    nn_float mfu;
    size_t vram_used_bytes;
    size_t vram_total_bytes;
} TrainStepStats;

// Activation functions
nn_float sigmoid(nn_float x);
nn_float sigmoid_derivative(nn_float activated, nn_float pre_activation);
nn_float relu(nn_float x);
nn_float relu_derivative(nn_float activated, nn_float pre_activation);
nn_float tanh_derivative(nn_float activated, nn_float pre_activation);

// Layer constructors
Layer *layer_dense_create(int input_size, int output_size);
Layer *layer_dense_create_backend(BackendKind backend, int input_size, int output_size);
Layer *layer_conv2d_create(int in_channels, int out_channels, int input_height,
                           int input_width, int kernel_h, int kernel_w, int stride_h,
                           int stride_w, int pad_h, int pad_w);
Layer *layer_conv2d_create_backend(BackendKind backend, int in_channels,
                                   int out_channels, int input_height, int input_width,
                                   int kernel_h, int kernel_w, int stride_h,
                                   int stride_w, int pad_h, int pad_w);
Layer *layer_activation_create(ActivationFn func, ActivationDerivativeFn derivative,
                               ActivationKind kind);
Layer *layer_tanh_create(void);
Layer *layer_dropout_create(nn_float rate);
void layer_dropout_set_training(Layer *layer, int training);
Layer *layer_activation_from_kind(ActivationKind kind);
Layer *layer_activation_from_kind_backend(BackendKind backend, ActivationKind kind);
Layer *layer_softmax_create(void);
Layer *layer_batchnorm_create(int channels, int spatial);
Layer *layer_batchnorm_create_backend(BackendKind backend, int channels, int spatial);
Layer *layer_batchnorm_create_conv(BackendKind backend, int channels, int height, int width);
void layer_batchnorm_set_training(Layer *layer, int training);
Layer *layer_maxpool_create(int channels, int input_height, int input_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
                            int pad_w);
Layer *layer_maxpool_create_backend(BackendKind backend, int channels, int input_height,
                                    int input_width, int kernel_h, int kernel_w, int stride_h,
                                    int stride_w, int pad_h, int pad_w);
Layer *layer_global_avgpool_create(int channels, int input_height, int input_width);
Layer *layer_global_avgpool_create_backend(BackendKind backend, int channels, int input_height,
                                           int input_width);
typedef struct {
    Layer *save;
    Layer *add;
} SkipConnection;
SkipConnection skip_connection_create(void);
void skip_connection_set_projection(SkipConnection *connection, Layer *projection_conv,
                                     Layer *projection_bn);
const char *activation_kind_name(ActivationKind kind);
const char *backend_kind_name(BackendKind kind);
const char *optimizer_kind_name(OptimizerKind kind);

// Neural network functions
Network *nn_create(int input_size, int hidden_size, int output_size);
Network *nn_create_empty(int input_size, int output_size, BackendKind backend);
Network *nn_create_mlp(int input_size, int hidden_size, int output_size,
                       ActivationKind hidden_activation,
                       ActivationKind output_activation, nn_float dropout_rate,
                       BackendKind backend);
Network *nn_create_with_backend(int input_size, int hidden_size, int output_size,
                                BackendKind backend);
void nn_add_layer(Network *nn, Layer *layer);
void nn_free(Network *nn);
void nn_print(Network *nn);
void nn_print_architecture(Network *nn);
Matrix *nn_forward(Network *nn, Matrix *input);
ForwardCache *nn_forward_cached(Network *nn, Matrix *input);
void forward_cache_free(ForwardCache *cache);
void nn_set_training(Network *nn, int training);
void nn_set_seed(Network *nn, unsigned long long seed);
void nn_set_gpu_theoretical_tflops(Network *nn, double tflops);
void nn_zero_gradients(Network *nn);
void nn_apply_gradients(Network *nn, nn_float learning_rate, int batch_size);
void nn_set_optimizer(Network *nn, OptimizerKind kind, nn_float beta1, nn_float beta2,
                      nn_float epsilon, nn_float weight_decay);
TrainStepStats nn_train_batch(Network *nn, Matrix *input, Matrix *target,
                              int batch_size, Matrix *output_buffer);
double nn_estimated_flops_per_sample(Network *nn);
void nn_set_log_memory(Network *nn, int log_memory);

// Training functions
BackwardCache *nn_backward(Network *nn, ForwardCache *forward_cache, Matrix *input,
                           Matrix *target);
void backward_cache_free(BackwardCache *cache);

#endif
