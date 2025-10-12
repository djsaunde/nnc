#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

#include "matrix.h"
#ifdef USE_CUDA
#include "gpu_backend.h"
#endif

typedef nn_float (*ActivationFn)(nn_float value);
typedef nn_float (*ActivationDerivativeFn)(nn_float activated, nn_float pre_activation);

typedef enum {
    LAYER_DENSE,
    LAYER_ACTIVATION,
    LAYER_DROPOUT,
    LAYER_SOFTMAX
} LayerType;

typedef enum {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    ACT_SOFTMAX
} ActivationKind;

typedef enum {
    BACKEND_CPU,
    BACKEND_GPU
} BackendKind;

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
} TrainStepStats;

// Activation functions
nn_float sigmoid(nn_float x);
nn_float sigmoid_derivative(nn_float activated, nn_float pre_activation);
nn_float relu(nn_float x);
nn_float relu_derivative(nn_float activated, nn_float pre_activation);
nn_float tanh_derivative(nn_float activated, nn_float pre_activation);

// Layer constructors
Layer *layer_dense_create(int input_size, int output_size);
Layer *layer_activation_create(ActivationFn func, ActivationDerivativeFn derivative);
Layer *layer_tanh_create(void);
Layer *layer_dropout_create(nn_float rate);
void layer_dropout_set_training(Layer *layer, int training);
Layer *layer_activation_from_kind(ActivationKind kind);
Layer *layer_softmax_create(void);
const char *activation_kind_name(ActivationKind kind);
const char *backend_kind_name(BackendKind kind);

// Neural network functions
Network *nn_create(int input_size, int hidden_size, int output_size);
Network *nn_create_mlp(int input_size, int hidden_size, int output_size,
                       ActivationKind hidden_activation,
                       ActivationKind output_activation, nn_float dropout_rate,
                       BackendKind backend);
Network *nn_create_with_backend(int input_size, int hidden_size, int output_size,
                                BackendKind backend);
void nn_add_layer(Network *nn, Layer *layer);
void nn_free(Network *nn);
void nn_print(Network *nn);
Matrix *nn_forward(Network *nn, Matrix *input);
ForwardCache *nn_forward_cached(Network *nn, Matrix *input);
void forward_cache_free(ForwardCache *cache);
void nn_set_training(Network *nn, int training);
void nn_zero_gradients(Network *nn);
void nn_apply_gradients(Network *nn, nn_float learning_rate, int batch_size);
TrainStepStats nn_train_batch(Network *nn, Matrix *input, Matrix *target, int batch_size,
                              Matrix *output_buffer);

// Training functions
BackwardCache *nn_backward(Network *nn, ForwardCache *forward_cache, Matrix *input,
                           Matrix *target);
void backward_cache_free(BackwardCache *cache);

#endif
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
