#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "gpu_backend.h"
#define CUDA_CALL(expr)                                                                \
    do {                                                                               \
        cudaError_t _cuda_status = (expr);                                             \
        if (_cuda_status != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                    cudaGetErrorString(_cuda_status));                                 \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)
#endif

#include "internals.h"
#include "matrix.h"

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

static inline int conv_output_dim(int input, int kernel, int stride, int padding)
{
    return (input + 2 * padding - kernel) / stride + 1;
}

static int conv_weight_index(const Conv2DLayerData *data, int oc, int ic, int kh,
                             int kw)
{
    int kernel_area = data->kernel_h * data->kernel_w;
    int weight_cols = data->in_channels * kernel_area;
    int offset = ic * kernel_area + kh * data->kernel_w + kw;
    return oc * weight_cols + offset;
}

Matrix *ensure_matrix(Matrix *matrix, int rows, int cols)
{
    if (matrix != NULL && matrix->rows == rows && matrix->cols == cols) {
        return matrix;
    }
    matrix_free(matrix);
    Matrix *result = matrix_create(rows, cols);
    matrix_zero(result);
    return result;
}

static Matrix *dense_forward_host(DenseLayerData *data, Matrix *input)
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, batch,
                input_size, alpha, data->weights->data, input_size, input->data, batch,
                beta, data->output_cache->data, batch);
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
    return dense_forward_host(data, input);
}

static Matrix *dense_backward_host(DenseLayerData *data, Matrix *grad_output)
{
    int rows = data->weights->rows;
    int cols = data->weights->cols;
    int batch = grad_output->cols;

#ifdef USE_CBLAS
    const nn_float alpha = 1.0f;
    const nn_float beta = 1.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, cols, batch, alpha,
                grad_output->data, grad_output->cols, data->last_input->data,
                data->last_input->cols, beta, data->grad_weights->data,
                data->grad_weights->cols);
#else
    for (int r = 0; r < rows; r++) {
        nn_float *grad_w_row = &data->grad_weights->data[r * cols];
        for (int b = 0; b < batch; b++) {
            nn_float grad_out = grad_output->data[r * grad_output->cols + b];
            for (int c = 0; c < cols; c++) {
                grad_w_row[c] +=
                    grad_out * data->last_input->data[c * data->last_input->cols + b];
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
                sum += data->weights->data[r * cols + c] *
                       grad_output->data[r * grad_output->cols + b];
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
    return dense_backward_host(data, grad_output);
}

void dense_layer_apply_sgd(DenseLayerData *data, nn_float learning_rate,
                           nn_float weight_decay)
{
    int total = data->weights->rows * data->weights->cols;
    if (weight_decay != 0.0f) {
#ifdef USE_CBLAS
        nn_float wd = weight_decay;
        cblas_saxpy(total, wd, data->weights->data, 1, data->grad_weights->data, 1);
#else
        for (int i = 0; i < total; i++) {
            data->grad_weights->data[i] += weight_decay * data->weights->data[i];
        }
#endif
    }
#ifdef USE_CBLAS
    nn_float neg_lr = -learning_rate;
    cblas_saxpy(total, neg_lr, data->grad_weights->data, 1, data->weights->data, 1);
    int bias_count = data->bias->rows * data->bias->cols;
    cblas_saxpy(bias_count, neg_lr, data->grad_bias->data, 1, data->bias->data, 1);
#else
    for (int i = 0; i < total; i++) {
        data->weights->data[i] -= learning_rate * data->grad_weights->data[i];
    }
    int bias_count = data->bias->rows * data->bias->cols;
    for (int i = 0; i < bias_count; i++) {
        data->bias->data[i] -= learning_rate * data->grad_bias->data[i];
    }
#endif
    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);
}

void dense_layer_apply_adamw(DenseLayerData *data, const OptimizerState *opt,
                             nn_float learning_rate, nn_float inv_bias_correction1,
                             nn_float inv_bias_correction2, nn_float grad_scale)
{
    int weight_rows = data->weights->rows;
    int weight_cols = data->weights->cols;
    int weight_total = weight_rows * weight_cols;
    int bias_rows = data->bias->rows;
    int bias_cols = data->bias->cols;
    int bias_total = bias_rows * bias_cols;

    data->adam_m_weights =
        ensure_matrix(data->adam_m_weights, weight_rows, weight_cols);
    data->adam_v_weights =
        ensure_matrix(data->adam_v_weights, weight_rows, weight_cols);
    data->adam_m_bias = ensure_matrix(data->adam_m_bias, bias_rows, bias_cols);
    data->adam_v_bias = ensure_matrix(data->adam_v_bias, bias_rows, bias_cols);

    if (!data->adam_initialized) {
        matrix_zero(data->adam_m_weights);
        matrix_zero(data->adam_v_weights);
        matrix_zero(data->adam_m_bias);
        matrix_zero(data->adam_v_bias);
        data->adam_initialized = 1;
    }

    nn_float one_minus_beta1 = 1.0f - opt->beta1;
    nn_float one_minus_beta2 = 1.0f - opt->beta2;

    for (int i = 0; i < weight_total; i++) {
        nn_float grad = data->grad_weights->data[i] * grad_scale;
        nn_float m = data->adam_m_weights->data[i] =
            opt->beta1 * data->adam_m_weights->data[i] + one_minus_beta1 * grad;
        nn_float v = data->adam_v_weights->data[i] =
            opt->beta2 * data->adam_v_weights->data[i] + one_minus_beta2 * grad * grad;
        nn_float m_hat = m * inv_bias_correction1;
        nn_float v_hat = v * inv_bias_correction2;
        nn_float denom = sqrtf(v_hat) + opt->epsilon;
        nn_float update = m_hat / denom;
        nn_float decay = opt->weight_decay > 0.0f
                             ? opt->weight_decay * data->weights->data[i]
                             : 0.0f;
        data->weights->data[i] -= learning_rate * (update + decay);
    }

    for (int i = 0; i < bias_total; i++) {
        nn_float grad = data->grad_bias->data[i] * grad_scale;
        nn_float m = data->adam_m_bias->data[i] =
            opt->beta1 * data->adam_m_bias->data[i] + one_minus_beta1 * grad;
        nn_float v = data->adam_v_bias->data[i] =
            opt->beta2 * data->adam_v_bias->data[i] + one_minus_beta2 * grad * grad;
        nn_float m_hat = m * inv_bias_correction1;
        nn_float v_hat = v * inv_bias_correction2;
        nn_float denom = sqrtf(v_hat) + opt->epsilon;
        nn_float update = m_hat / denom;
        data->bias->data[i] -= learning_rate * update;
    }

    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);
}

static void dense_update(Layer *layer, nn_float learning_rate)
{
    DenseLayerData *data = (DenseLayerData *) layer->data;

    if (data->backend == BACKEND_GPU) {
        return;
    }

    dense_layer_apply_sgd(data, learning_rate, 0.0f);
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
    matrix_free(data->adam_m_weights);
    matrix_free(data->adam_v_weights);
    matrix_free(data->adam_m_bias);
    matrix_free(data->adam_v_bias);
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU && data->gpu != NULL) {
        dense_gpu_destroy(data->gpu);
    }
#endif
    free(data);
}

static Matrix *conv2d_forward(Layer *layer, Matrix *input)
{
    Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
    int batch = input->cols;
    int input_size = data->in_channels * data->input_height * data->input_width;

    if (input->rows != input_size) {
        fprintf(stderr, "Conv2D forward dimension mismatch\n");
        return NULL;
    }

    data->last_input = ensure_matrix(data->last_input, input_size, batch);
    matrix_copy_into(data->last_input, input);

    int output_rows = data->out_channels * data->output_height * data->output_width;
    data->output_cache = ensure_matrix(data->output_cache, output_rows, batch);
    matrix_zero(data->output_cache);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < data->out_channels; oc++) {
            for (int oh = 0; oh < data->output_height; oh++) {
                for (int ow = 0; ow < data->output_width; ow++) {
                    nn_float sum = data->bias->data[oc];
                    for (int ic = 0; ic < data->in_channels; ic++) {
                        for (int kh = 0; kh < data->kernel_h; kh++) {
                            for (int kw = 0; kw < data->kernel_w; kw++) {
                                int ih = oh * data->stride_h - data->pad_h + kh;
                                int iw = ow * data->stride_w - data->pad_w + kw;
                                if (ih < 0 || ih >= data->input_height || iw < 0 ||
                                    iw >= data->input_width) {
                                    continue;
                                }
                                int input_row = ((ic * data->input_height + ih) *
                                                     data->input_width +
                                                 iw);
                                nn_float input_val =
                                    data->last_input->data[input_row * batch + b];
                                int weight_index =
                                    conv_weight_index(data, oc, ic, kh, kw);
                                nn_float weight_val = data->weights->data[weight_index];
                                sum += weight_val * input_val;
                            }
                        }
                    }
                    int output_row =
                        ((oc * data->output_height + oh) * data->output_width + ow);
                    data->output_cache->data[output_row * batch + b] = sum;
                }
            }
        }
    }

    return matrix_copy(data->output_cache);
}

static Matrix *conv2d_backward(Layer *layer, Matrix *grad_output)
{
    Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
    int batch = grad_output->cols;
    int output_rows = data->out_channels * data->output_height * data->output_width;
    if (grad_output->rows != output_rows) {
        fprintf(stderr, "Conv2D backward dimension mismatch\n");
        return NULL;
    }

    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);

    int input_rows = data->in_channels * data->input_height * data->input_width;
    Matrix *grad_input = matrix_create(input_rows, batch);
    matrix_zero(grad_input);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < data->out_channels; oc++) {
            for (int oh = 0; oh < data->output_height; oh++) {
                for (int ow = 0; ow < data->output_width; ow++) {
                    int output_row =
                        ((oc * data->output_height + oh) * data->output_width + ow);
                    nn_float grad_out =
                        grad_output->data[output_row * grad_output->cols + b];
                    data->grad_bias->data[oc] += grad_out;

                    for (int ic = 0; ic < data->in_channels; ic++) {
                        for (int kh = 0; kh < data->kernel_h; kh++) {
                            for (int kw = 0; kw < data->kernel_w; kw++) {
                                int ih = oh * data->stride_h - data->pad_h + kh;
                                int iw = ow * data->stride_w - data->pad_w + kw;
                                if (ih < 0 || ih >= data->input_height || iw < 0 ||
                                    iw >= data->input_width) {
                                    continue;
                                }
                                int input_row = ((ic * data->input_height + ih) *
                                                     data->input_width +
                                                 iw);
                                nn_float input_val =
                                    data->last_input
                                        ->data[input_row * data->last_input->cols + b];
                                int weight_index =
                                    conv_weight_index(data, oc, ic, kh, kw);
                                data->grad_weights->data[weight_index] +=
                                    grad_out * input_val;
                                nn_float weight_val = data->weights->data[weight_index];
                                grad_input->data[input_row * grad_input->cols + b] +=
                                    grad_out * weight_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void conv2d_layer_apply_sgd(Conv2DLayerData *data, nn_float learning_rate,
                            nn_float weight_decay)
{
    int total = data->weights->rows * data->weights->cols;
    if (weight_decay != 0.0f) {
        for (int i = 0; i < total; i++) {
            data->grad_weights->data[i] += weight_decay * data->weights->data[i];
        }
    }
    for (int i = 0; i < total; i++) {
        data->weights->data[i] -= learning_rate * data->grad_weights->data[i];
    }
    int bias_total = data->bias->rows * data->bias->cols;
    for (int i = 0; i < bias_total; i++) {
        data->bias->data[i] -= learning_rate * data->grad_bias->data[i];
    }
    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);
}

void conv2d_layer_apply_adamw(Conv2DLayerData *data, const OptimizerState *opt,
                              nn_float learning_rate, nn_float inv_bias_correction1,
                              nn_float inv_bias_correction2, nn_float grad_scale)
{
    int weight_rows = data->weights->rows;
    int weight_cols = data->weights->cols;
    int weight_total = weight_rows * weight_cols;
    int bias_rows = data->bias->rows;
    int bias_cols = data->bias->cols;
    int bias_total = bias_rows * bias_cols;

    data->adam_m_weights =
        ensure_matrix(data->adam_m_weights, weight_rows, weight_cols);
    data->adam_v_weights =
        ensure_matrix(data->adam_v_weights, weight_rows, weight_cols);
    data->adam_m_bias = ensure_matrix(data->adam_m_bias, bias_rows, bias_cols);
    data->adam_v_bias = ensure_matrix(data->adam_v_bias, bias_rows, bias_cols);

    if (!data->adam_initialized) {
        matrix_zero(data->adam_m_weights);
        matrix_zero(data->adam_v_weights);
        matrix_zero(data->adam_m_bias);
        matrix_zero(data->adam_v_bias);
        data->adam_initialized = 1;
    }

    nn_float one_minus_beta1 = 1.0f - opt->beta1;
    nn_float one_minus_beta2 = 1.0f - opt->beta2;

    for (int i = 0; i < weight_total; i++) {
        nn_float grad = data->grad_weights->data[i] * grad_scale;
        nn_float m = data->adam_m_weights->data[i] =
            opt->beta1 * data->adam_m_weights->data[i] + one_minus_beta1 * grad;
        nn_float v = data->adam_v_weights->data[i] =
            opt->beta2 * data->adam_v_weights->data[i] + one_minus_beta2 * grad * grad;
        nn_float m_hat = m * inv_bias_correction1;
        nn_float v_hat = v * inv_bias_correction2;
        nn_float denom = sqrtf(v_hat) + opt->epsilon;
        nn_float update = m_hat / denom;
        nn_float decay = opt->weight_decay > 0.0f
                             ? opt->weight_decay * data->weights->data[i]
                             : 0.0f;
        data->weights->data[i] -= learning_rate * (update + decay);
    }

    for (int i = 0; i < bias_total; i++) {
        nn_float grad = data->grad_bias->data[i] * grad_scale;
        nn_float m = data->adam_m_bias->data[i] =
            opt->beta1 * data->adam_m_bias->data[i] + one_minus_beta1 * grad;
        nn_float v = data->adam_v_bias->data[i] =
            opt->beta2 * data->adam_v_bias->data[i] + one_minus_beta2 * grad * grad;
        nn_float m_hat = m * inv_bias_correction1;
        nn_float v_hat = v * inv_bias_correction2;
        nn_float denom = sqrtf(v_hat) + opt->epsilon;
        nn_float update = m_hat / denom;
        data->bias->data[i] -= learning_rate * update;
    }

    matrix_zero(data->grad_weights);
    matrix_zero(data->grad_bias);
}

static void conv2d_update(Layer *layer, nn_float learning_rate)
{
    Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
    if (data->backend == BACKEND_GPU) {
        return;
    }
    conv2d_layer_apply_sgd(data, learning_rate, 0.0f);
}

static void conv2d_destroy(Layer *layer)
{
    Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
    matrix_free(data->weights);
    matrix_free(data->bias);
    matrix_free(data->last_input);
    matrix_free(data->output_cache);
    matrix_free(data->grad_weights);
    matrix_free(data->grad_bias);
    matrix_free(data->adam_m_weights);
    matrix_free(data->adam_v_weights);
    matrix_free(data->adam_m_bias);
    matrix_free(data->adam_v_bias);
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU && data->gpu != NULL) {
        conv2d_gpu_destroy(data->gpu);
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
#ifdef USE_CUDA
    data->d_last_output = NULL;
#endif
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
#ifdef USE_CUDA
    if (data->d_mask != NULL) {
        CUDA_CALL(cudaFree(data->d_mask));
    }
#endif
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
#ifdef USE_CUDA
    data->d_last_output = NULL;
#endif
    free(data);
}

Layer *layer_conv2d_create(int in_channels, int out_channels, int input_height,
                           int input_width, int kernel_h, int kernel_w, int stride_h,
                           int stride_w, int pad_h, int pad_w)
{
    if (in_channels <= 0 || out_channels <= 0 || input_height <= 0 ||
        input_width <= 0 || kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 ||
        stride_w <= 0 || pad_h < 0 || pad_w < 0) {
        fprintf(stderr, "Invalid Conv2D parameters.\n");
        return NULL;
    }

    int out_h_numerator = input_height + 2 * pad_h - kernel_h;
    int out_w_numerator = input_width + 2 * pad_w - kernel_w;
    if (out_h_numerator < 0 || out_w_numerator < 0) {
        fprintf(stderr, "Conv2D output shape is invalid with given parameters.\n");
        return NULL;
    }

    int output_height = conv_output_dim(input_height, kernel_h, stride_h, pad_h);
    int output_width = conv_output_dim(input_width, kernel_w, stride_w, pad_w);
    if (output_height <= 0 || output_width <= 0) {
        fprintf(stderr, "Conv2D output dimensions must be positive.\n");
        return NULL;
    }

    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    Conv2DLayerData *data = (Conv2DLayerData *) calloc(1, sizeof(Conv2DLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->in_channels = in_channels;
    data->out_channels = out_channels;
    data->input_height = input_height;
    data->input_width = input_width;
    data->kernel_h = kernel_h;
    data->kernel_w = kernel_w;
    data->stride_h = stride_h;
    data->stride_w = stride_w;
    data->pad_h = pad_h;
    data->pad_w = pad_w;
    data->output_height = output_height;
    data->output_width = output_width;
    data->backend = BACKEND_CPU;

    int weight_rows = out_channels;
    int weight_cols = in_channels * kernel_h * kernel_w;
    data->weights = matrix_create(weight_rows, weight_cols);
    matrix_randomize(data->weights);
    matrix_scale(data->weights, sqrt(1.0 / (in_channels * kernel_h * kernel_w)));

    data->bias = matrix_create(out_channels, 1);
    matrix_fill(data->bias, 0.0f);

    data->last_input = NULL;
    data->output_cache = NULL;
    data->grad_weights = matrix_create(weight_rows, weight_cols);
    matrix_zero(data->grad_weights);
    data->grad_bias = matrix_create(out_channels, 1);
    matrix_zero(data->grad_bias);
    data->adam_m_weights = NULL;
    data->adam_v_weights = NULL;
    data->adam_m_bias = NULL;
    data->adam_v_bias = NULL;
    data->adam_initialized = 0;
#ifdef USE_CUDA
    data->adam_device_initialized = 0;
    data->gpu = NULL;
#endif

    layer->type = LAYER_CONV2D;
    layer->data = data;
    layer->forward = conv2d_forward;
    layer->backward = conv2d_backward;
    layer->update = conv2d_update;
    layer->destroy = conv2d_destroy;

    return layer;
}

Layer *layer_conv2d_create_backend(BackendKind backend, int in_channels,
                                   int out_channels, int input_height, int input_width,
                                   int kernel_h, int kernel_w, int stride_h,
                                   int stride_w, int pad_h, int pad_w)
{
    if (backend == BACKEND_CPU) {
        return layer_conv2d_create(in_channels, out_channels, input_height, input_width,
                                   kernel_h, kernel_w, stride_h, stride_w, pad_h,
                                   pad_w);
    }

#ifdef USE_CUDA
    Layer *layer =
        layer_conv2d_create(in_channels, out_channels, input_height, input_width,
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    if (layer == NULL) {
        return NULL;
    }
    Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
    data->backend = BACKEND_GPU;
    data->gpu = conv2d_gpu_create(in_channels, out_channels, input_height, input_width,
                                  kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    if (data->gpu == NULL) {
        layer->destroy(layer);
        free(layer);
        return NULL;
    }
    conv2d_gpu_set_weights(data->gpu, data->weights->data, data->bias->data);
    conv2d_gpu_zero_gradients(data->gpu);
    return layer;
#else
    (void) in_channels;
    (void) out_channels;
    (void) input_height;
    (void) input_width;
    (void) kernel_h;
    (void) kernel_w;
    (void) stride_h;
    (void) stride_w;
    (void) pad_h;
    (void) pad_w;
    fprintf(stderr, "Conv2D layer with GPU backend requires CUDA support.\n");
    return NULL;
#endif
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
    data->adam_m_weights = NULL;
    data->adam_v_weights = NULL;
    data->adam_m_bias = NULL;
    data->adam_v_bias = NULL;
    data->adam_initialized = 0;
#ifdef USE_CUDA
    data->gpu = NULL;
    data->adam_device_initialized = 0;
#endif

    layer->type = LAYER_DENSE;
    layer->data = data;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->update = dense_update;
    layer->destroy = dense_destroy;

    return layer;
}

Layer *layer_activation_create(ActivationFn func, ActivationDerivativeFn derivative,
                               ActivationKind kind)
{
    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    ActivationLayerData *data =
        (ActivationLayerData *) malloc(sizeof(ActivationLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->func = func;
    data->derivative = derivative;
    data->kind = kind;
    data->last_input = NULL;
    data->last_output = NULL;
#ifdef USE_CUDA
    data->d_last_output = NULL;
    data->last_rows = 0;
    data->last_cols = 0;
#endif
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
    return layer_activation_create(nn_tanh, tanh_derivative, ACT_TANH);
}

Layer *layer_activation_from_kind(ActivationKind kind)
{
    switch (kind) {
    case ACT_SIGMOID:
        return layer_activation_create(sigmoid, sigmoid_derivative, ACT_SIGMOID);
    case ACT_RELU:
        return layer_activation_create(relu, relu_derivative, ACT_RELU);
    case ACT_TANH:
        return layer_activation_create(nn_tanh, tanh_derivative, ACT_TANH);
    case ACT_SOFTMAX:
        return layer_softmax_create();
    }
    return NULL;
}

Layer *layer_activation_from_kind_backend(BackendKind backend, ActivationKind kind)
{
    (void) backend;
    return layer_activation_from_kind(kind);
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
#ifdef USE_CUDA
    data->d_mask = NULL;
    data->mask_capacity = 0;
#endif

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
#ifdef USE_CUDA
    data->d_last_output = NULL;
    data->last_rows = 0;
    data->last_cols = 0;
#endif

    layer->type = LAYER_SOFTMAX;
    layer->data = data;
    layer->forward = softmax_forward;
    layer->backward = softmax_backward;
    layer->update = softmax_update;
    layer->destroy = softmax_destroy;

    return layer;
}

static Matrix *batchnorm_forward(Layer *layer, Matrix *input)
{
    BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
    int channels = data->channels;
    int spatial = data->spatial;
    int cols = input->cols;
    int rows = input->rows;
    if (rows != channels * spatial) {
        fprintf(stderr, "BatchNorm forward dimension mismatch.\n");
        return matrix_copy(input);
    }

    int elements_per_channel = spatial * cols;
    if (elements_per_channel <= 0) {
        return matrix_copy(input);
    }

    data->input_cache = ensure_matrix(data->input_cache, rows, cols);
    matrix_copy_into(data->input_cache, input);
    data->normalized = ensure_matrix(data->normalized, rows, cols);
    data->batch_mean = ensure_matrix(data->batch_mean, channels, 1);
    data->batch_var = ensure_matrix(data->batch_var, channels, 1);

    Matrix *output = matrix_create(rows, cols);

    nn_float *gamma = data->gamma->data;
    nn_float *beta = data->beta->data;
    nn_float *running_mean = data->running_mean->data;
    nn_float *running_var = data->running_var->data;
    nn_float *batch_mean = data->batch_mean->data;
    nn_float *batch_var = data->batch_var->data;

    const nn_float epsilon = data->epsilon;
    const nn_float momentum = data->momentum;

    if (data->training) {
        for (int c = 0; c < channels; c++) {
            nn_float sum = 0.0f;
            nn_float sumsq = 0.0f;
            for (int s = 0; s < spatial; s++) {
                int row = c * spatial + s;
                nn_float *row_ptr = &input->data[row * cols];
                for (int col = 0; col < cols; col++) {
                    nn_float v = row_ptr[col];
                    sum += v;
                    sumsq += v * v;
                }
            }
            nn_float mean = sum / (nn_float) elements_per_channel;
            nn_float var = sumsq / (nn_float) elements_per_channel - mean * mean;
            if (var < 0.0f) {
                var = 0.0f;
            }
            batch_mean[c] = mean;
            batch_var[c] = var;
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;

            nn_float inv_std = 1.0f / sqrtf(var + epsilon);
            for (int s = 0; s < spatial; s++) {
                int row = c * spatial + s;
                nn_float *row_ptr = &input->data[row * cols];
                nn_float *norm_row = &data->normalized->data[row * cols];
                nn_float *out_row = &output->data[row * cols];
                for (int col = 0; col < cols; col++) {
                    nn_float norm = (row_ptr[col] - mean) * inv_std;
                    norm_row[col] = norm;
                    out_row[col] = gamma[c] * norm + beta[c];
                }
            }
        }
    } else {
        for (int c = 0; c < channels; c++) {
            nn_float mean = running_mean[c];
            nn_float var = running_var[c];
            nn_float inv_std = 1.0f / sqrtf(var + epsilon);
            for (int s = 0; s < spatial; s++) {
                int row = c * spatial + s;
                nn_float *row_ptr = &input->data[row * cols];
                nn_float *out_row = &output->data[row * cols];
                for (int col = 0; col < cols; col++) {
                    nn_float norm = (row_ptr[col] - mean) * inv_std;
                    out_row[col] = gamma[c] * norm + beta[c];
                }
            }
        }
    }

    return output;
}

static Matrix *batchnorm_backward(Layer *layer, Matrix *grad_output)
{
    BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
    if (!data->training) {
        return matrix_copy(grad_output);
    }

    int channels = data->channels;
    int spatial = data->spatial;
    int cols = grad_output->cols;
    int rows = grad_output->rows;
    int elements_per_channel = spatial * cols;
    if (rows != channels * spatial || elements_per_channel <= 0) {
        return matrix_copy(grad_output);
    }

    Matrix *grad_input = matrix_create(rows, cols);

    nn_float *gamma = data->gamma->data;
    nn_float *beta_grad = data->grad_beta->data;
    nn_float *gamma_grad = data->grad_gamma->data;
    nn_float *batch_var = data->batch_var->data;
    const nn_float epsilon = data->epsilon;

    for (int c = 0; c < channels; c++) {
        nn_float sum_dy = 0.0f;
        nn_float sum_dy_norm = 0.0f;
        for (int s = 0; s < spatial; s++) {
            int row = c * spatial + s;
            nn_float *grad_row = &grad_output->data[row * cols];
            nn_float *norm_row = &data->normalized->data[row * cols];
            for (int col = 0; col < cols; col++) {
                nn_float dy = grad_row[col];
                nn_float norm = norm_row[col];
                sum_dy += dy;
                sum_dy_norm += dy * norm;
            }
        }
        beta_grad[c] += sum_dy;
        gamma_grad[c] += sum_dy_norm;

        nn_float inv_std = 1.0f / sqrtf(batch_var[c] + epsilon);
        nn_float scale = gamma[c] * inv_std;
        for (int s = 0; s < spatial; s++) {
            int row = c * spatial + s;
            nn_float *grad_row = &grad_output->data[row * cols];
            nn_float *norm_row = &data->normalized->data[row * cols];
            nn_float *out_row = &grad_input->data[row * cols];
            for (int col = 0; col < cols; col++) {
                nn_float dy = grad_row[col];
                nn_float norm = norm_row[col];
                nn_float term = (nn_float) elements_per_channel * dy - sum_dy - norm * sum_dy_norm;
                out_row[col] = scale * term / (nn_float) elements_per_channel;
            }
        }
    }

    return grad_input;
}

void batchnorm_layer_zero_gradients(BatchNormLayerData *data)
{
    if (data == NULL) {
        return;
    }
    if (data->grad_gamma != NULL) {
        matrix_zero(data->grad_gamma);
    }
    if (data->grad_beta != NULL) {
        matrix_zero(data->grad_beta);
    }
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        size_t bytes = (size_t) data->channels * sizeof(nn_float);
        batchnorm_layer_gpu_copy_params_to_device(data);
        batchnorm_gpu_zero_gradients(data->d_grad_gamma, data->d_grad_beta, data->channels);
        if (data->d_m_gamma != NULL && data->adam_m_gamma != NULL) {
            CUDA_CALL(cudaMemcpy(data->d_m_gamma, data->adam_m_gamma->data, bytes,
                                 cudaMemcpyHostToDevice));
        }
        if (data->d_v_gamma != NULL && data->adam_v_gamma != NULL) {
            CUDA_CALL(cudaMemcpy(data->d_v_gamma, data->adam_v_gamma->data, bytes,
                                 cudaMemcpyHostToDevice));
        }
        if (data->d_m_beta != NULL && data->adam_m_beta != NULL) {
            CUDA_CALL(cudaMemcpy(data->d_m_beta, data->adam_m_beta->data, bytes,
                                 cudaMemcpyHostToDevice));
        }
        if (data->d_v_beta != NULL && data->adam_v_beta != NULL) {
            CUDA_CALL(cudaMemcpy(data->d_v_beta, data->adam_v_beta->data, bytes,
                                 cudaMemcpyHostToDevice));
        }
    }
#endif
}

static void batchnorm_destroy(Layer *layer)
{
    if (layer == NULL) {
        return;
    }
    BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
    if (data != NULL) {
        matrix_free(data->gamma);
        matrix_free(data->beta);
        matrix_free(data->running_mean);
        matrix_free(data->running_var);
        matrix_free(data->batch_mean);
        matrix_free(data->batch_var);
        matrix_free(data->normalized);
        matrix_free(data->input_cache);
        matrix_free(data->grad_gamma);
        matrix_free(data->grad_beta);
        matrix_free(data->adam_m_gamma);
        matrix_free(data->adam_v_gamma);
        matrix_free(data->adam_m_beta);
        matrix_free(data->adam_v_beta);
    }
#ifdef USE_CUDA
        if (data->backend == BACKEND_GPU) {
            if (data->d_gamma != NULL) {
                CUDA_CALL(cudaFree(data->d_gamma));
            }
            if (data->d_beta != NULL) {
                CUDA_CALL(cudaFree(data->d_beta));
            }
            if (data->d_running_mean != NULL) {
                CUDA_CALL(cudaFree(data->d_running_mean));
            }
            if (data->d_running_var != NULL) {
                CUDA_CALL(cudaFree(data->d_running_var));
            }
            if (data->d_batch_mean != NULL) {
                CUDA_CALL(cudaFree(data->d_batch_mean));
            }
            if (data->d_batch_var != NULL) {
                CUDA_CALL(cudaFree(data->d_batch_var));
            }
            if (data->d_normalized != NULL) {
                CUDA_CALL(cudaFree(data->d_normalized));
            }
            if (data->d_grad_input != NULL) {
                CUDA_CALL(cudaFree(data->d_grad_input));
            }
            if (data->d_grad_gamma != NULL) {
                CUDA_CALL(cudaFree(data->d_grad_gamma));
            }
            if (data->d_grad_beta != NULL) {
                CUDA_CALL(cudaFree(data->d_grad_beta));
            }
            if (data->d_m_gamma != NULL) {
                CUDA_CALL(cudaFree(data->d_m_gamma));
            }
            if (data->d_v_gamma != NULL) {
                CUDA_CALL(cudaFree(data->d_v_gamma));
            }
            if (data->d_m_beta != NULL) {
                CUDA_CALL(cudaFree(data->d_m_beta));
            }
            if (data->d_v_beta != NULL) {
                CUDA_CALL(cudaFree(data->d_v_beta));
            }
        }
#endif
    free(data);
    free(layer);
}

static Layer *layer_batchnorm_create_internal(int channels, int spatial,
                                              BackendKind backend)
{
    if (channels <= 0 || spatial <= 0) {
        return NULL;
    }

    Layer *layer = (Layer *) malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    BatchNormLayerData *data = (BatchNormLayerData *) calloc(1, sizeof(BatchNormLayerData));
    if (data == NULL) {
        free(layer);
        return NULL;
    }

    data->channels = channels;
    data->spatial = spatial;
    data->epsilon = 1e-5f;
    data->momentum = 0.1f;
    data->training = 1;
    data->backend = backend;

    data->gamma = matrix_create(channels, 1);
    data->beta = matrix_create(channels, 1);
    data->running_mean = matrix_create(channels, 1);
    data->running_var = matrix_create(channels, 1);
    data->grad_gamma = matrix_create(channels, 1);
    data->grad_beta = matrix_create(channels, 1);
    data->adam_m_gamma = matrix_create(channels, 1);
    data->adam_v_gamma = matrix_create(channels, 1);
    data->adam_m_beta = matrix_create(channels, 1);
    data->adam_v_beta = matrix_create(channels, 1);
    if (data->gamma == NULL || data->beta == NULL || data->running_mean == NULL ||
        data->running_var == NULL || data->grad_gamma == NULL || data->grad_beta == NULL ||
        data->adam_m_gamma == NULL || data->adam_v_gamma == NULL || data->adam_m_beta == NULL ||
        data->adam_v_beta == NULL) {
        batchnorm_destroy(layer);
        return NULL;
    }

    for (int c = 0; c < channels; c++) {
        data->gamma->data[c] = 1.0f;
        data->beta->data[c] = 0.0f;
        data->running_mean->data[c] = 0.0f;
        data->running_var->data[c] = 1.0f;
    }
    matrix_zero(data->grad_gamma);
    matrix_zero(data->grad_beta);
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        batchnorm_gpu_zero_gradients(data->d_grad_gamma, data->d_grad_beta, data->channels);
    }
#endif
    matrix_zero(data->adam_m_gamma);
    matrix_zero(data->adam_v_gamma);
    matrix_zero(data->adam_m_beta);
    matrix_zero(data->adam_v_beta);

#ifdef USE_CUDA
    data->d_gamma = NULL;
    data->d_beta = NULL;
    data->d_running_mean = NULL;
    data->d_running_var = NULL;
    data->d_batch_mean = NULL;
    data->d_batch_var = NULL;
    data->d_normalized = NULL;
    data->d_input_cache = NULL;
    data->d_grad_input = NULL;
    data->d_grad_gamma = NULL;
    data->d_grad_beta = NULL;
#endif

    layer->type = LAYER_BATCHNORM;
    layer->data = data;
    layer->forward = batchnorm_forward;
    layer->backward = batchnorm_backward;
    layer->update = NULL;
    layer->destroy = batchnorm_destroy;
    return layer;
}

Layer *layer_batchnorm_create(int channels, int spatial)
{
    return layer_batchnorm_create_internal(channels, spatial, BACKEND_CPU);
}

Layer *layer_batchnorm_create_backend(BackendKind backend, int channels, int spatial)
{
    Layer *layer = layer_batchnorm_create_internal(channels, spatial, backend);
    if (layer == NULL) {
        return NULL;
    }
#ifdef USE_CUDA
    if (backend == BACKEND_GPU) {
        BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
        size_t bytes = (size_t) channels * sizeof(nn_float);
        CUDA_CALL(cudaMalloc((void **) &data->d_gamma, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_beta, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_running_mean, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_running_var, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_batch_mean, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_batch_var, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_grad_gamma, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_grad_beta, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_m_gamma, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_v_gamma, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_m_beta, bytes));
        CUDA_CALL(cudaMalloc((void **) &data->d_v_beta, bytes));
        batchnorm_layer_gpu_copy_params_to_device(data);
        batchnorm_layer_gpu_copy_running_to_device(data);
        CUDA_CALL(cudaMemset(data->d_grad_gamma, 0, bytes));
        CUDA_CALL(cudaMemset(data->d_grad_beta, 0, bytes));
        CUDA_CALL(cudaMemset(data->d_m_gamma, 0, bytes));
        CUDA_CALL(cudaMemset(data->d_v_gamma, 0, bytes));
        CUDA_CALL(cudaMemset(data->d_m_beta, 0, bytes));
        CUDA_CALL(cudaMemset(data->d_v_beta, 0, bytes));
        data->gpu_capacity = 0;
        data->adam_device_initialized = 1;
    }
#else
    if (backend == BACKEND_GPU) {
        fprintf(stderr, "BatchNorm GPU backend not available; using CPU implementation.\n");
    }
#endif
    return layer;
}

Layer *layer_batchnorm_create_conv(BackendKind backend, int channels, int height, int width)
{
    return layer_batchnorm_create_backend(backend, channels, height * width);
}

#ifdef USE_CUDA
void batchnorm_layer_gpu_ensure_capacity(BatchNormLayerData *data, int batch)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    if (batch <= 0) {
        return;
    }
    int rows = data->channels * data->spatial;
    size_t elements = (size_t) rows * (size_t) batch;
    if (batch > data->gpu_capacity) {
        if (data->d_normalized != NULL) {
            CUDA_CALL(cudaFree(data->d_normalized));
        }
        if (data->d_grad_input != NULL) {
            CUDA_CALL(cudaFree(data->d_grad_input));
        }
        CUDA_CALL(cudaMalloc((void **) &data->d_normalized,
                             elements * sizeof(nn_float)));
        CUDA_CALL(cudaMalloc((void **) &data->d_grad_input,
                             elements * sizeof(nn_float)));
        data->gpu_capacity = batch;
    }
}

void batchnorm_layer_gpu_copy_params_to_device(BatchNormLayerData *data)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    size_t bytes = (size_t) data->channels * sizeof(nn_float);
    CUDA_CALL(cudaMemcpy(data->d_gamma, data->gamma->data, bytes,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data->d_beta, data->beta->data, bytes,
                         cudaMemcpyHostToDevice));
}

void batchnorm_layer_gpu_copy_params_to_host(BatchNormLayerData *data)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    size_t bytes = (size_t) data->channels * sizeof(nn_float);
    CUDA_CALL(cudaMemcpy(data->gamma->data, data->d_gamma, bytes,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(data->beta->data, data->d_beta, bytes,
                         cudaMemcpyDeviceToHost));
}

void batchnorm_layer_gpu_copy_running_to_device(BatchNormLayerData *data)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    size_t bytes = (size_t) data->channels * sizeof(nn_float);
    CUDA_CALL(cudaMemcpy(data->d_running_mean, data->running_mean->data, bytes,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(data->d_running_var, data->running_var->data, bytes,
                         cudaMemcpyHostToDevice));
}

void batchnorm_layer_gpu_copy_running_to_host(BatchNormLayerData *data)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    size_t bytes = (size_t) data->channels * sizeof(nn_float);
    CUDA_CALL(cudaMemcpy(data->running_mean->data, data->d_running_mean, bytes,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(data->running_var->data, data->d_running_var, bytes,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(data->batch_mean->data, data->d_batch_mean, bytes,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(data->batch_var->data, data->d_batch_var, bytes,
                         cudaMemcpyDeviceToHost));
}

void batchnorm_layer_gpu_copy_grads_to_host(BatchNormLayerData *data)
{
    if (data == NULL || data->backend != BACKEND_GPU) {
        return;
    }
    size_t bytes = (size_t) data->channels * sizeof(nn_float);
    CUDA_CALL(cudaMemcpy(data->grad_gamma->data, data->d_grad_gamma, bytes,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(data->grad_beta->data, data->d_grad_beta, bytes,
                         cudaMemcpyDeviceToHost));
}
#endif

void layer_batchnorm_set_training(Layer *layer, int training)
{
    if (layer == NULL || layer->type != LAYER_BATCHNORM) {
        return;
    }
    BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
    data->training = training ? 1 : 0;
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        batchnorm_layer_gpu_copy_params_to_device(data);
        batchnorm_layer_gpu_copy_running_to_device(data);
    }
#endif
}

void batchnorm_layer_apply_sgd(BatchNormLayerData *data, nn_float learning_rate,
                               nn_float weight_decay, nn_float grad_scale)
{
    if (data == NULL) {
        return;
    }
#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        batchnorm_gpu_apply_sgd(data->d_gamma, data->d_beta, data->d_grad_gamma,
                                data->d_grad_beta, data->channels, learning_rate,
                                weight_decay, grad_scale);
        matrix_zero(data->grad_gamma);
        matrix_zero(data->grad_beta);
        batchnorm_layer_gpu_copy_params_to_host(data);
        return;
    }
#endif
    int channels = data->channels;
    nn_float *gamma = data->gamma->data;
    nn_float *beta = data->beta->data;
    nn_float *grad_gamma = data->grad_gamma->data;
    nn_float *grad_beta = data->grad_beta->data;

    for (int c = 0; c < channels; c++) {
        nn_float g_gamma = grad_gamma[c] * grad_scale;
        nn_float g_beta = grad_beta[c] * grad_scale;
        if (weight_decay != 0.0f) {
            g_gamma += weight_decay * gamma[c];
        }
        gamma[c] -= learning_rate * g_gamma;
        beta[c] -= learning_rate * g_beta;
    }

    matrix_zero(data->grad_gamma);
    matrix_zero(data->grad_beta);
}

void batchnorm_layer_apply_adamw(BatchNormLayerData *data, const OptimizerState *opt,
                                 nn_float learning_rate, nn_float inv_bias_correction1,
                                 nn_float inv_bias_correction2, nn_float grad_scale)
{
    if (data == NULL) {
        return;
    }

#ifdef USE_CUDA
    if (data->backend == BACKEND_GPU) {
        if (!data->adam_device_initialized) {
            size_t bytes = (size_t) data->channels * sizeof(nn_float);
            CUDA_CALL(cudaMemset(data->d_m_gamma, 0, bytes));
            CUDA_CALL(cudaMemset(data->d_v_gamma, 0, bytes));
            CUDA_CALL(cudaMemset(data->d_m_beta, 0, bytes));
            CUDA_CALL(cudaMemset(data->d_v_beta, 0, bytes));
            data->adam_device_initialized = 1;
        }
        batchnorm_gpu_apply_adamw(data->d_gamma, data->d_beta, data->d_grad_gamma,
                                   data->d_grad_beta, data->d_m_gamma, data->d_v_gamma,
                                   data->d_m_beta, data->d_v_beta, data->channels,
                                   learning_rate, opt->beta1, opt->beta2, opt->epsilon,
                                   opt->weight_decay, inv_bias_correction1,
                                   inv_bias_correction2, grad_scale);
        matrix_zero(data->grad_gamma);
        matrix_zero(data->grad_beta);
        data->adam_initialized = 1;
        batchnorm_layer_gpu_copy_params_to_host(data);
        return;
    }
#endif

    int channels = data->channels;
    data->adam_m_gamma = ensure_matrix(data->adam_m_gamma, channels, 1);
    data->adam_v_gamma = ensure_matrix(data->adam_v_gamma, channels, 1);
    data->adam_m_beta = ensure_matrix(data->adam_m_beta, channels, 1);
    data->adam_v_beta = ensure_matrix(data->adam_v_beta, channels, 1);

    if (!data->adam_initialized) {
        matrix_zero(data->adam_m_gamma);
        matrix_zero(data->adam_v_gamma);
        matrix_zero(data->adam_m_beta);
        matrix_zero(data->adam_v_beta);
        data->adam_initialized = 1;
    }

    nn_float one_minus_beta1 = 1.0f - opt->beta1;
    nn_float one_minus_beta2 = 1.0f - opt->beta2;

    nn_float *gamma = data->gamma->data;
    nn_float *beta = data->beta->data;
    nn_float *grad_gamma = data->grad_gamma->data;
    nn_float *grad_beta = data->grad_beta->data;

    for (int c = 0; c < channels; c++) {
        nn_float g_gamma = grad_gamma[c] * grad_scale;
        nn_float g_beta = grad_beta[c] * grad_scale;

        nn_float m_gamma =
            data->adam_m_gamma->data[c] = opt->beta1 * data->adam_m_gamma->data[c] +
                                          one_minus_beta1 * g_gamma;
        nn_float v_gamma =
            data->adam_v_gamma->data[c] = opt->beta2 * data->adam_v_gamma->data[c] +
                                          one_minus_beta2 * g_gamma * g_gamma;

        nn_float m_beta =
            data->adam_m_beta->data[c] = opt->beta1 * data->adam_m_beta->data[c] +
                                         one_minus_beta1 * g_beta;
        nn_float v_beta =
            data->adam_v_beta->data[c] = opt->beta2 * data->adam_v_beta->data[c] +
                                         one_minus_beta2 * g_beta * g_beta;

        nn_float m_hat_gamma = m_gamma * inv_bias_correction1;
        nn_float v_hat_gamma = v_gamma * inv_bias_correction2;
        nn_float denom_gamma = sqrtf(v_hat_gamma) + opt->epsilon;
        nn_float update_gamma = m_hat_gamma / denom_gamma;
        if (opt->weight_decay != 0.0f) {
            update_gamma += opt->weight_decay * gamma[c];
        }
        gamma[c] -= learning_rate * update_gamma;

        nn_float m_hat_beta = m_beta * inv_bias_correction1;
        nn_float v_hat_beta = v_beta * inv_bias_correction2;
        nn_float denom_beta = sqrtf(v_hat_beta) + opt->epsilon;
        nn_float update_beta = m_hat_beta / denom_beta;
        beta[c] -= learning_rate * update_beta;
    }

    matrix_zero(data->grad_gamma);
    matrix_zero(data->grad_beta);
}

static void skip_connection_release(SkipConnectionData *data)
{
    if (data == NULL) {
        return;
    }
    if (--data->refcount > 0) {
        return;
    }
    matrix_free(data->saved_input);
    matrix_free(data->grad_accum);
#ifdef USE_CUDA
    if (data->d_saved_input != NULL) {
        CUDA_CALL(cudaFree(data->d_saved_input));
    }
    if (data->d_grad_accum != NULL) {
        CUDA_CALL(cudaFree(data->d_grad_accum));
    }
#endif
    free(data);
}

static Matrix *skip_save_forward(Layer *layer, Matrix *input)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    data->saved_input = ensure_matrix(data->saved_input, input->rows, input->cols);
    matrix_copy_into(data->saved_input, input);
    if (data->grad_accum != NULL && (data->grad_accum->rows != input->rows ||
                                     data->grad_accum->cols != input->cols)) {
        matrix_free(data->grad_accum);
        data->grad_accum = NULL;
    }
    return matrix_copy(input);
}

static Matrix *skip_save_backward(Layer *layer, Matrix *grad_output)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    Matrix *result = matrix_copy(grad_output);
    if (data->grad_accum != NULL) {
        matrix_add_inplace(result, data->grad_accum);
        matrix_zero(data->grad_accum);
    }
    return result;
}

static void skip_save_update(Layer *layer, nn_float learning_rate)
{
    (void) layer;
    (void) learning_rate;
}

static void skip_save_destroy(Layer *layer)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    skip_connection_release(data);
}

static Matrix *skip_add_forward(Layer *layer, Matrix *input)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    if (data->saved_input == NULL || data->saved_input->rows != input->rows ||
        data->saved_input->cols != input->cols) {
        return matrix_copy(input);
    }

    Matrix *output = matrix_copy(input);
    int total = output->rows * output->cols;
#ifdef USE_CBLAS
    nn_float alpha = 1.0f;
    cblas_saxpy(total, alpha, data->saved_input->data, 1, output->data, 1);
#else
    for (int i = 0; i < total; i++) {
        output->data[i] += data->saved_input->data[i];
    }
#endif
    return output;
}

static Matrix *skip_add_backward(Layer *layer, Matrix *grad_output)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    data->grad_accum =
        ensure_matrix(data->grad_accum, grad_output->rows, grad_output->cols);
    matrix_copy_into(data->grad_accum, grad_output);
    return matrix_copy(grad_output);
}

static void skip_add_update(Layer *layer, nn_float learning_rate)
{
    (void) layer;
    (void) learning_rate;
}

static void skip_add_destroy(Layer *layer)
{
    SkipConnectionData *data = (SkipConnectionData *) layer->data;
    skip_connection_release(data);
}

Layer *layer_dense_create_backend(BackendKind backend, int input_size, int output_size)
{
    switch (backend) {
    case BACKEND_CPU:
        return layer_dense_create(input_size, output_size);
    case BACKEND_GPU: {
#ifdef USE_CUDA
        Layer *layer = layer_dense_create(input_size, output_size);
        if (layer == NULL) {
            return NULL;
        }
        DenseLayerData *data = (DenseLayerData *) layer->data;
        data->backend = BACKEND_GPU;
        data->weights_dirty = 0;
        data->gpu = dense_gpu_create(input_size, output_size);
        if (data->gpu == NULL) {
            layer->destroy(layer);
            free(layer);
            return NULL;
        }
        dense_gpu_set_weights(data->gpu, data->weights->data, data->bias->data);
        dense_gpu_zero_gradients(data->gpu);
        return layer;
#else
        fprintf(stderr, "GPU backend requested but USE_CUDA is not enabled.\n");
        return NULL;
#endif
    }
    }
    return NULL;
}

SkipConnection skip_connection_create(void)
{
    SkipConnection result = {0};
    SkipConnectionData *data =
        (SkipConnectionData *) calloc(1, sizeof(SkipConnectionData));
    if (data == NULL) {
        return result;
    }
    data->refcount = 2;

    Layer *save = (Layer *) calloc(1, sizeof(Layer));
    Layer *add = (Layer *) calloc(1, sizeof(Layer));
    if (save == NULL || add == NULL) {
        free(save);
        free(add);
        skip_connection_release(data);
        return result;
    }

    save->type = LAYER_SKIP_SAVE;
    save->data = data;
    save->forward = skip_save_forward;
    save->backward = skip_save_backward;
    save->update = skip_save_update;
    save->destroy = skip_save_destroy;

    add->type = LAYER_SKIP_ADD;
    add->data = data;
    add->forward = skip_add_forward;
    add->backward = skip_add_backward;
    add->update = skip_add_update;
    add->destroy = skip_add_destroy;

    result.save = save;
    result.add = add;
    return result;
}

void skip_connection_set_projection(SkipConnection *connection, Layer *projection_conv,
                                     Layer *projection_bn)
{
    if (connection == NULL || connection->save == NULL) {
        return;
    }
    SkipConnectionData *data = (SkipConnectionData *) connection->save->data;
    data->projection_conv = projection_conv;
    data->projection_bn = projection_bn;
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

const char *optimizer_kind_name(OptimizerKind kind)
{
    switch (kind) {
    case OPTIMIZER_SGD:
        return "sgd";
    case OPTIMIZER_ADAMW:
        return "adamw";
    }
    return "unknown";
}
