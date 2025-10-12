#include "net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_CBLAS
#include <cblas.h>
#endif
#ifdef USE_CUDA
#include "gpu_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

typedef struct {
    Matrix *weights;
    Matrix *bias;
    Matrix *last_input;
    Matrix *grad_weights;
    Matrix *grad_bias;
    Matrix *output_cache;
#ifdef USE_CUDA
    DenseLayerGpuContext *gpu;
#endif
    BackendKind backend;
    int weights_dirty;
} DenseLayerData;

typedef struct {
    ActivationFn func;
    ActivationDerivativeFn derivative;
    Matrix *last_input;
    Matrix *last_output;
} ActivationLayerData;

typedef struct {
    nn_float rate;
    nn_float keep_probability;
    int training;
    Matrix *mask;
} DropoutLayerData;

typedef struct {
    Matrix *last_input;
    Matrix *last_output;
} SoftmaxLayerData;

static Matrix *ensure_matrix(Matrix *matrix, int rows, int cols)
{
    if (matrix == NULL || matrix->rows != rows || matrix->cols != cols) {
        if (matrix != NULL) {
            matrix_free(matrix);
        }
        matrix = matrix_create(rows, cols);
        matrix_zero(matrix);
    }
    return matrix;
}

nn_float sigmoid(nn_float x)
{
    return (nn_float) (1.0f / (1.0f + expf(-x)));
}

nn_float sigmoid_derivative(nn_float activated, nn_float pre_activation)
{
    (void) pre_activation;
    return activated * (1.0f - activated);
}

nn_float relu(nn_float x)
{
    return x > 0 ? x : 0.0f;
}

nn_float relu_derivative(nn_float activated, nn_float pre_activation)
{
    (void) activated;
    return pre_activation > 0 ? 1.0f : 0.0f;
}

nn_float tanh_derivative(nn_float activated, nn_float pre_activation)
{
    (void) pre_activation;
    return 1.0f - activated * activated;
}

static nn_float nn_tanh(nn_float x)
{
    return tanhf(x);
}

static Matrix *dense_forward_cpu(DenseLayerData *data, Matrix *input)
{
    int input_size = data->weights->cols;
    int output_size = data->weights->rows;
    int batch = input->cols;

    if (input->rows != input_size) {
        printf("Dense forward dimension mismatch\n");
        return NULL;
    }

    data->last_input = ensure_matrix(data->last_input, input_size, batch);
    matrix_copy_into(data->last_input, input);

    data->output_cache = ensure_matrix(data->output_cache, output_size, batch);

#ifdef USE_CBLAS
    const nn_float alpha = 1.0f;
    const nn_float beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, batch, input_size, alpha,
                data->weights->data, input_size, input->data, batch, beta,
                data->output_cache->data, batch);
#else
    for (int col = 0; col < batch; col++) {
        for (int row = 0; row < output_size; row++) {
            nn_float sum = 0.0f;
            nn_float *w_row = &data->weights->data[row * input_size];
            for (int k = 0; k < input_size; k++) {
                sum += w_row[k] * input->data[k * input->cols + col];
            }
            data->output_cache->data[row * batch + col] = sum;
        }
    }
#endif

    for (int col = 0; col < batch; col++) {
        for (int row = 0; row < output_size; row++) {
            data->output_cache->data[row * batch + col] += data->bias->data[row];
        }
    }

    return matrix_copy(data->output_cache);
}

static Matrix *dense_forward(Layer *layer, Matrix *input)
{
    DenseLayerData *data = (DenseLayerData *) layer->data;
    if (data->backend == BACKEND_CPU) {
        return dense_forward_cpu(data, input);
    }
#ifdef USE_CUDA
    int input_size = data->weights->cols;
    int output_size = data->weights->rows;
    int batch = input->cols;

    if (input->rows != input_size) {
        fprintf(stderr, "Dense forward dimension mismatch\n");
        return NULL;
    }

    data->last_input = ensure_matrix(data->last_input, input_size, batch);
    matrix_copy_into(data->last_input, input);

    data->output_cache = ensure_matrix(data->output_cache, output_size, batch);

    if (data->weights_dirty) {
        dense_gpu_update(data->gpu, data->weights->data, data->bias->data);
        data->weights_dirty = 0;
    }

    dense_gpu_forward(data->gpu, input->data, batch, data->output_cache->data,
                      data->last_input->data);

    for (int col = 0; col < batch; col++) {
        for (int row = 0; row < output_size; row++) {
            data->output_cache->data[row * batch + col] += data->bias->data[row];
        }
    }

    return matrix_copy(data->output_cache);
#else
    fprintf(stderr, "GPU support not enabled. Recompile with USE_CUDA=1.\n");
    return NULL;
#endif
}

static Matrix *dense_backward_cpu(DenseLayerData *data, Matrix *grad_output)
{
    int rows = data->weights->rows;
    int cols = data->weights->cols;
    int batch = grad_output->cols;

#ifdef USE_CBLAS
    const nn_float alpha = 1.0f;
    const nn_float beta = 1.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, cols, batch, alpha,
                grad_output->data, grad_output->cols, data->last_input->data,
                data->last_input->cols, beta, data->grad_weights->data, data->grad_weights->cols);
#else
    for (int r = 0; r < rows; r++) {
        nn_float *grad_w_row = &data->grad_weights->data[r * cols];
        for (int b = 0; b < batch; b++) {
            nn_float grad_out = grad_output->data[r * grad_output->cols + b];
            for (int c = 0; c < cols; c++) {
                grad_w_row[c] += grad_out * data->last_input->data[c * data->last_input->cols + b];
            }
        }
    }
#endif

    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < rows; r++) {
            data->grad_bias->data[r] += grad_output->data[r * grad_output->cols + b];
        }
    }

    Matrix *grad_input = matrix_create(cols, batch);
#ifdef USE_CBLAS
    const nn_float alpha2 = 1.0f;
    const nn_float beta2 = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, cols, batch, rows, alpha2,
                data->weights->data, cols, grad_output->data, grad_output->cols, beta2,
                grad_input->data, grad_input->cols);
#else
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < cols; c++) {
            nn_float sum = 0.0f;
            for (int r = 0; r < rows; r++) {
                sum += data->weights->data[r * cols + c] * grad_output->data[r * grad_output->cols + b];
            }
            grad_input->data[c * grad_input->cols + b] = sum;
        }
    }
#endif

    return grad_input;
}

static Matrix *dense_backward(Layer *layer, Matrix *grad_output)
{
    DenseLayerData *data = (DenseLayerData *) layer->data;
    if (data->backend == BACKEND_CPU) {
        return dense_backward_cpu(data, grad_output);
    }
#ifdef USE_CUDA
    int rows = data->weights->rows;
    int cols = data->weights->cols;
    int batch = grad_output->cols;

    Matrix *grad_input = matrix_create(cols, batch);
    dense_gpu_backward(data->gpu, grad_output->data, data->last_input->data, batch,
                       data->grad_weights->data, grad_input->data);

    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < rows; r++) {
            data->grad_bias->data[r] += grad_output->data[r * grad_output->cols + b];
        }
    }

    return grad_input;
#else
    fprintf(stderr, "GPU support not enabled. Recompile with USE_CUDA=1.\n");
    return NULL;
#endif
}

static void dense_update(Layer *layer, nn_float learning_rate)
{
    DenseLayerData *data = (DenseLayerData *) layer->data;

    int total = data->weights->rows * data->weights->cols;
#ifdef USE_CBLAS
    cblas_saxpy(total, -learning_rate, data->grad_weights->data, 1, data->weights->data, 1);
    cblas_saxpy(data->bias->rows, -learning_rate, data->grad_bias->data, 1, data->bias->data, 1);
#else
    for (int i = 0; i < total; i++) {
        data->weights->data[i] -= learning_rate * data->grad_weights->data[i];
    }
    for (int i = 0; i < data->bias->rows; i++) {
        data->bias->data[i] -= learning_rate * data->grad_bias->data[i];
    }
#endif

#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        dense_gpu_update(data->gpu, data->weights->data, data->bias->data);
        data->weights_dirty = 0;
    }
#endif

    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);
}

static void dense_destroy(Layer *layer)
{
    DenseLayerData *data = (DenseLayerData *) layer->data;
    matrix_free(data->weights);
    matrix_free(data->bias);
    matrix_free(data->last_input);
    matrix_free(data->grad_weights);
    matrix_free(data->grad_bias);
    matrix_free(data->output_cache);
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU && data->gpu != NULL) {
        dense_gpu_destroy(data->gpu);
    }
#endif
    free(data);
}

static Matrix *activation_forward(Layer *layer, Matrix *input)
{
    ActivationLayerData *data = (ActivationLayerData *) layer->data;

    if (data->last_input == NULL || data->last_input->rows != input->rows ||
        data->last_input->cols != input->cols) {
        if (data->last_input != NULL) {
            matrix_free(data->last_input);
        }
        data->last_input = matrix_create(input->rows, input->cols);
    }
    matrix_copy_into(data->last_input, input);

    if (data->last_output == NULL || data->last_output->rows != input->rows ||
        data->last_output->cols != input->cols) {
        if (data->last_output != NULL) {
            matrix_free(data->last_output);
        }
        data->last_output = matrix_create(input->rows, input->cols);
    }
    matrix_copy_into(data->last_output, input);
    matrix_apply(data->last_output, data->func);

    return matrix_copy(data->last_output);
}

static Matrix *activation_backward(Layer *layer, Matrix *grad_output)
{
    ActivationLayerData *data = (ActivationLayerData *) layer->data;
    Matrix *grad_input = matrix_create(data->last_input->rows, data->last_input->cols);
    int total = data->last_input->rows * data->last_input->cols;

    for (int i = 0; i < total; i++) {
        nn_float activated = data->last_output->data[i];
        nn_float raw = data->last_input->data[i];
        nn_float derivative = data->derivative(activated, raw);
        grad_input->data[i] = grad_output->data[i] * derivative;
    }

    return grad_input;
}

static void activation_update(Layer *layer, nn_float learning_rate)
{
    (void) layer;
    (void) learning_rate;
}

static void activation_destroy(Layer *layer)
{
    ActivationLayerData *data = (ActivationLayerData *) layer->data;
    if (data->last_input != NULL) {
        matrix_free(data->last_input);
    }
    if (data->last_output != NULL) {
        matrix_free(data->last_output);
    }
    free(data);
}

static Matrix *dropout_forward(Layer *layer, Matrix *input)
{
    DropoutLayerData *data = (DropoutLayerData *) layer->data;

    if (!data->training) {
        if (data->mask != NULL) {
            matrix_free(data->mask);
            data->mask = NULL;
        }
        return matrix_copy(input);
    }

    if (data->mask == NULL || data->mask->rows != input->rows ||
        data->mask->cols != input->cols) {
        if (data->mask != NULL) {
            matrix_free(data->mask);
        }
        data->mask = matrix_create(input->rows, input->cols);
    }

    Matrix *output = matrix_copy(input);
    nn_float keep = data->keep_probability;
    int total = input->rows * input->cols;
    for (int i = 0; i < total; i++) {
        nn_float sample = (nn_float) rand() / (nn_float) RAND_MAX;
        nn_float retain = sample >= data->rate ? 1.0f : 0.0f;
        nn_float scale = retain / keep;
        data->mask->data[i] = scale;
        output->data[i] *= scale;
    }

    return output;
}

static Matrix *dropout_backward(Layer *layer, Matrix *grad_output)
{
    DropoutLayerData *data = (DropoutLayerData *) layer->data;
    Matrix *grad_input = matrix_copy(grad_output);

    if (!data->training || data->mask == NULL) {
        return grad_input;
    }

    int total = grad_output->rows * grad_output->cols;
    for (int i = 0; i < total; i++) {
        grad_input->data[i] *= data->mask->data[i];
    }

    return grad_input;
}

static void dropout_update(Layer *layer, nn_float learning_rate)
{
    (void) layer;
    (void) learning_rate;
}

static void dropout_destroy(Layer *layer)
{
    DropoutLayerData *data = (DropoutLayerData *) layer->data;
    if (data->mask != NULL) {
        matrix_free(data->mask);
    }
    free(data);
}

static Matrix *softmax_forward(Layer *layer, Matrix *input)
{
    SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;

    data->last_input = ensure_matrix(data->last_input, input->rows, input->cols);
    matrix_copy_into(data->last_input, input);

    data->last_output = ensure_matrix(data->last_output, input->rows, input->cols);
    matrix_copy_into(data->last_output, input);

    int rows = input->rows;
    int cols = input->cols;

    for (int col = 0; col < cols; col++) {
        nn_float max_value = data->last_output->data[col];
        for (int row = 1; row < rows; row++) {
            nn_float value = data->last_output->data[row * cols + col];
            if (value > max_value) {
                max_value = value;
            }
        }

        nn_float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            nn_float value = data->last_output->data[row * cols + col];
            value = expf(value - max_value);
            data->last_output->data[row * cols + col] = value;
            sum += value;
        }

        if (sum == 0.0f) {
            sum = 1.0f;
        }
        for (int row = 0; row < rows; row++) {
            data->last_output->data[row * cols + col] /= sum;
        }
    }

    return matrix_copy(data->last_output);
}

static Matrix *softmax_backward(Layer *layer, Matrix *grad_output)
{
    SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;
    int rows = data->last_output->rows;
    int cols = data->last_output->cols;
    Matrix *grad_input = matrix_create(rows, cols);

    for (int col = 0; col < cols; col++) {
        for (int i = 0; i < rows; i++) {
            nn_float yi = data->last_output->data[i * cols + col];
            nn_float sum = 0.0f;
            for (int j = 0; j < rows; j++) {
                nn_float yj = data->last_output->data[j * cols + col];
                nn_float derivative = (i == j) ? yi * (1.0f - yi) : -yi * yj;
                sum += derivative * grad_output->data[j * grad_output->cols + col];
            }
            grad_input->data[i * cols + col] = sum;
        }
    }

    return grad_input;
}

static void softmax_update(Layer *layer, nn_float learning_rate)
{
    (void) layer;
    (void) learning_rate;
}

static void softmax_destroy(Layer *layer)
{
    SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;
    if (data->last_input != NULL) {
        matrix_free(data->last_input);
    }
    if (data->last_output != NULL) {
        matrix_free(data->last_output);
    }
    free(data);
}

Layer *layer_dense_create(int input_size, int output_size)
{
    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    DenseLayerData *data = (DenseLayerData *) malloc(sizeof(DenseLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->weights = matrix_create(output_size, input_size);
    matrix_randomize(data->weights);
    matrix_scale(data->weights, sqrt(1.0 / input_size));

    data->bias = matrix_create(output_size, 1);
    matrix_fill(data->bias, 0.0);

    data->last_input = matrix_create(input_size, 1);
    matrix_zero(data->last_input);
    data->grad_weights = matrix_create(output_size, input_size);
    matrix_zero(data->grad_weights);
    data->grad_bias = matrix_create(output_size, 1);
    matrix_zero(data->grad_bias);
    data->output_cache = matrix_create(output_size, 1);
    matrix_zero(data->output_cache);
    data->backend = BACKEND_CPU;
    data->weights_dirty = 0;
#ifdef USE_CUDA
    data->gpu = NULL;
#endif

    layer->type = LAYER_DENSE;
    layer->data = data;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    layer->destroy = dense_destroy;

    return layer;
}

static Layer *layer_dense_create_backend(BackendKind backend, int input_size, int output_size)
{
    switch (backend) {
    case BACKEND_CPU:
        return layer_dense_create(input_size, output_size);
    case BACKEND_GPU:
    {
#ifdef USE_CUDA
        Layer *layer = layer_dense_create(input_size, output_size);
        if (layer == NULL) {
            return NULL;
        }
        DenseLayerData *data = (DenseLayerData *) layer->data;
        data->backend = BACKEND_GPU;
        data->weights_dirty = 1;
        data->gpu = dense_gpu_create(input_size, output_size);
        if (data->gpu == NULL) {
            layer->destroy(layer);
            free(layer);
            return NULL;
        }
        dense_gpu_set_weights(data->gpu, data->weights->data, data->bias->data);
        data->weights_dirty = 0;
        return layer;
#else
        fprintf(stderr, "GPU backend requested but USE_CUDA is not enabled.\n");
        return NULL;
#endif
    }
    }
    return NULL;
}

Layer *layer_activation_create(ActivationFn func, ActivationDerivativeFn derivative)
{
    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    ActivationLayerData *data = (ActivationLayerData *) malloc(sizeof(ActivationLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->func = func;
    data->derivative = derivative;
    data->last_input = NULL;
    data->last_output = NULL;

    layer->type = LAYER_ACTIVATION;
    layer->data = data;
    layer->forward = activation_forward;
    layer->backward = activation_backward;
    layer->update = activation_update;
    layer->destroy = activation_destroy;

    return layer;
}

Layer *layer_tanh_create(void)
{
    return layer_activation_create(nn_tanh, tanh_derivative);
}

Layer *layer_activation_from_kind(ActivationKind kind)
{
    switch (kind) {
    case ACT_SIGMOID:
        return layer_activation_create(sigmoid, sigmoid_derivative);
    case ACT_RELU:
        return layer_activation_create(relu, relu_derivative);
    case ACT_TANH:
        return layer_activation_create(nn_tanh, tanh_derivative);
    case ACT_SOFTMAX:
        return layer_softmax_create();
    }
    return NULL;
}

static Layer *layer_activation_from_kind_backend(BackendKind backend, ActivationKind kind)
{
    (void) backend;
    return layer_activation_from_kind(kind);
}

const char *activation_kind_name(ActivationKind kind)
{
    switch (kind) {
    case ACT_SIGMOID:
        return "sigmoid";
    case ACT_RELU:
        return "relu";
    case ACT_TANH:
        return "tanh";
    case ACT_SOFTMAX:
        return "softmax";
    }
    return "unknown";
}

const char *backend_kind_name(BackendKind kind)
{
    switch (kind) {
    case BACKEND_CPU:
        return "CPU";
    case BACKEND_GPU:
        return "GPU";
    }
    return "unknown";
}

Layer *layer_softmax_create(void)
{
    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    SoftmaxLayerData *data = (SoftmaxLayerData *) malloc(sizeof(SoftmaxLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }
    data->last_input = NULL;
    data->last_output = NULL;

    layer->type = LAYER_SOFTMAX;
    layer->data = data;
    layer->forward = softmax_forward;
    layer->backward = softmax_backward;
    layer->update = softmax_update;
    layer->destroy = softmax_destroy;

    return layer;
}

Layer *layer_dropout_create(nn_float rate)
{
    if (rate < 0.0f || rate >= 1.0f) {
        return NULL;
    }

    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    DropoutLayerData *data = (DropoutLayerData *) malloc(sizeof(DropoutLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->rate = rate;
    data->keep_probability = 1.0f - rate;
    data->training = 1;
    data->mask = NULL;

    layer->type = LAYER_DROPOUT;
    layer->data = data;
    layer->forward = dropout_forward;
    layer->backward = dropout_backward;
    layer->update = dropout_update;
    layer->destroy = dropout_destroy;

    return layer;
}

void layer_dropout_set_training(Layer *layer, int training)
{
    if (layer == NULL || layer->type != LAYER_DROPOUT) {
        return;
    }
    DropoutLayerData *data = (DropoutLayerData *) layer->data;
    data->training = training ? 1 : 0;
}

static void nn_destroy_layer(Layer *layer)
{
    if (layer != NULL) {
        layer->destroy(layer);
        free(layer);
    }
}

Network *nn_create_mlp(int input_size, int hidden_size, int output_size,
                       ActivationKind hidden_activation,
                       ActivationKind output_activation, nn_float dropout_rate,
                       BackendKind backend)
{
    Network *nn = (Network *) malloc(sizeof(Network));
    if (nn == NULL) {
        return NULL;
    }

    nn->layer_count = 0;
    nn->capacity = 4;
    nn->layers = (Layer **) malloc(nn->capacity * sizeof(Layer *));
    if (nn->layers == NULL) {
        free(nn);
        return NULL;
    }
    nn->input_size = input_size;
    nn->output_size = output_size;
    nn->backend = backend;

    Layer *dense_ih = layer_dense_create_backend(backend, input_size, hidden_size);
    Layer *act_hidden = layer_activation_from_kind_backend(backend, hidden_activation);
    Layer *dropout = NULL;
    if (dropout_rate > 0.0f) {
        dropout = layer_dropout_create(dropout_rate);
    }
    Layer *dense_ho = layer_dense_create_backend(backend, hidden_size, output_size);
    Layer *act_output = layer_activation_from_kind_backend(backend, output_activation);

    if (dense_ih == NULL || act_hidden == NULL || dense_ho == NULL ||
        act_output == NULL || (dropout_rate > 0.0f && dropout == NULL)) {
        nn_destroy_layer(dense_ih);
        nn_destroy_layer(act_hidden);
        nn_destroy_layer(dropout);
        nn_destroy_layer(dense_ho);
        nn_destroy_layer(act_output);
        free(nn->layers);
        free(nn);
        return NULL;
    }

    nn_add_layer(nn, dense_ih);
    nn_add_layer(nn, act_hidden);
    if (dropout != NULL) {
        nn_add_layer(nn, dropout);
    }
    nn_add_layer(nn, dense_ho);
    nn_add_layer(nn, act_output);

    return nn;
}

Network *nn_create(int input_size, int hidden_size, int output_size)
{
    return nn_create_mlp(input_size, hidden_size, output_size, ACT_SIGMOID, ACT_SIGMOID,
                         0.0, BACKEND_CPU);
}

Network *nn_create_with_backend(int input_size, int hidden_size, int output_size,
                                BackendKind backend)
{
    return nn_create_mlp(input_size, hidden_size, output_size, ACT_SIGMOID, ACT_SIGMOID, 0.0,
                         backend);
}

void nn_add_layer(Network *nn, Layer *layer)
{
    if (layer == NULL) {
        return;
    }

    if (nn->layer_count == nn->capacity) {
        size_t new_capacity = nn->capacity == 0 ? 1 : nn->capacity * 2;
        Layer **new_layers =
            (Layer **) realloc(nn->layers, new_capacity * sizeof(Layer *));
        if (new_layers == NULL) {
            return;
        }
        nn->layers = new_layers;
        nn->capacity = new_capacity;
    }

    nn->layers[nn->layer_count++] = layer;
}

void nn_free(Network *nn)
{
    if (nn == NULL) {
        return;
    }

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        if (layer != NULL) {
            layer->destroy(layer);
            free(layer);
        }
    }

    free(nn->layers);
    free(nn);
}

void nn_print(Network *nn)
{
    printf("Neural Network Architecture:\n");
    printf("  Layers: %zu\n", nn->layer_count);

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            printf("  [%zu] Dense: %d -> %d\n", i, data->weights->cols, data->weights->rows);
            break;
        }
        case LAYER_ACTIVATION: {
            const char *name = "activation";
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            if (data->func == sigmoid) {
                name = "sigmoid";
            } else if (data->func == relu) {
                name = "relu";
            } else if (data->func == nn_tanh) {
                name = "tanh";
            }
            printf("  [%zu] Activation: %s\n", i, name);
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            printf("  [%zu] Dropout: rate=%.2f training=%s\n", i, data->rate,
                   data->training ? "yes" : "no");
            break;
        }
        case LAYER_SOFTMAX:
            printf("  [%zu] Softmax\n", i);
            break;
        }
    }
}

Matrix *nn_forward(Network *nn, Matrix *input)
{
    ForwardCache *cache = nn_forward_cached(nn, input);
    Matrix *output = matrix_copy(cache->output);
    forward_cache_free(cache);
    return output;
}

ForwardCache *nn_forward_cached(Network *nn, Matrix *input)
{
    ForwardCache *cache = (ForwardCache *) malloc(sizeof(ForwardCache));
    Matrix *current = matrix_copy(input);

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        Matrix *next = layer->forward(layer, current);
        matrix_free(current);
        current = next;
    }

    cache->output = current;
    return cache;
}

void forward_cache_free(ForwardCache *cache)
{
    if (cache == NULL) {
        return;
    }
    if (cache->output != NULL) {
        matrix_free(cache->output);
    }
    free(cache);
}

BackwardCache *nn_backward(Network *nn, ForwardCache *forward_cache, Matrix *input,
                           Matrix *target)
{
    (void) input;

    BackwardCache *backward_cache = (BackwardCache *) malloc(sizeof(BackwardCache));

    const nn_float epsilon = 1e-7f;
    int output_rows = forward_cache->output->rows;
    int batch = forward_cache->output->cols;
    Layer *output_layer = nn->layer_count > 0 ? nn->layers[nn->layer_count - 1] : NULL;
    int use_softmax = output_layer != NULL && output_layer->type == LAYER_SOFTMAX;

    backward_cache->loss = 0.0f;
    if (use_softmax) {
        for (int col = 0; col < batch; col++) {
            for (int row = 0; row < output_rows; row++) {
                nn_float y = target->data[row * target->cols + col];
                if (y <= 0.0) {
                    continue;
                }
                nn_float y_hat = forward_cache->output->data[row * forward_cache->output->cols + col];
                y_hat = fmaxf(epsilon, y_hat);
                backward_cache->loss += -y * logf(y_hat);
            }
        }
        backward_cache->loss /= batch;
    } else if (output_rows == 1) {
        for (int col = 0; col < batch; col++) {
            nn_float y = target->data[col];
            nn_float y_hat = forward_cache->output->data[col];
            y_hat = fmaxf(epsilon, fminf(1.0f - epsilon, y_hat));
            backward_cache->loss += -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
        }
        backward_cache->loss /= batch;
    } else {
        for (int col = 0; col < batch; col++) {
            for (int row = 0; row < output_rows; row++) {
                nn_float y = target->data[row * target->cols + col];
                nn_float y_hat = forward_cache->output->data[row * forward_cache->output->cols + col];
                y_hat = fmaxf(epsilon, fminf(1.0f - epsilon, y_hat));
                backward_cache->loss += -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
            }
        }
        backward_cache->loss /= (output_rows * batch);
    }

    backward_cache->output_gradient = matrix_subtract(forward_cache->output, target);

    backward_cache->grad_norm = 0.0f;
    int total = backward_cache->output_gradient->rows * backward_cache->output_gradient->cols;
    for (int i = 0; i < total; i++) {
        nn_float value = backward_cache->output_gradient->data[i];
        backward_cache->grad_norm += value * value;
    }
    backward_cache->grad_norm = sqrtf(backward_cache->grad_norm) / (nn_float) batch;

    Matrix *current_grad = backward_cache->output_gradient;
    for (size_t idx = nn->layer_count; idx-- > 0;) {
        Layer *layer = nn->layers[idx];
        Matrix *next_grad = layer->backward(layer, current_grad);
        if (current_grad != backward_cache->output_gradient) {
            matrix_free(current_grad);
        }
        current_grad = next_grad;
    }

    backward_cache->input_gradient = current_grad;

    return backward_cache;
}

void backward_cache_free(BackwardCache *cache)
{
    if (cache == NULL) {
        return;
    }
    if (cache->output_gradient != NULL) {
        matrix_free(cache->output_gradient);
    }
    if (cache->input_gradient != NULL) {
        matrix_free(cache->input_gradient);
    }
    free(cache);
}

void nn_zero_gradients(Network *nn)
{
    if (nn == NULL) {
        return;
    }
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        if (layer->type == LAYER_DENSE) {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            matrix_zero(data->grad_weights);
            matrix_zero(data->grad_bias);
        }
    }
}

void nn_apply_gradients(Network *nn, nn_float learning_rate, int batch_size)
{
    if (nn == NULL) {
        return;
    }
    if (batch_size <= 0) {
        batch_size = 1;
    }
    nn_float scaled_lr = learning_rate / (nn_float) batch_size;
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        layer->update(layer, scaled_lr);
    }
}

TrainStepStats nn_train_batch(Network *nn, Matrix *input, Matrix *target, int batch_size,
                              Matrix *output_buffer)
{
    TrainStepStats stats = {.loss = 0.0f, .grad_norm = 0.0f, .samples = batch_size};

    ForwardCache *forward_cache = nn_forward_cached(nn, input);

    if (output_buffer != NULL && output_buffer->rows == forward_cache->output->rows &&
        output_buffer->cols >= batch_size) {
        for (int row = 0; row < forward_cache->output->rows; row++) {
            for (int col = 0; col < batch_size; col++) {
                output_buffer->data[row * output_buffer->cols + col] =
                    forward_cache->output->data[row * forward_cache->output->cols + col];
            }
        }
    }

    BackwardCache *backward_cache = nn_backward(nn, forward_cache, input, target);

    stats.loss = backward_cache->loss;
    stats.grad_norm = backward_cache->grad_norm;

    backward_cache_free(backward_cache);
    forward_cache_free(forward_cache);

    return stats;
}

void nn_set_training(Network *nn, int training)
{
    if (nn == NULL) {
        return;
    }
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        if (layer->type == LAYER_DROPOUT) {
            layer_dropout_set_training(layer, training);
        }
    }
}
