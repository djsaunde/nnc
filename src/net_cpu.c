#include <math.h>
#include <stdlib.h>

#include "dataset.h"
#include "net_internal.h"
#include "matrix.h"

ForwardCache *nn_forward_cached(Network *nn, Matrix *input)
{
#ifdef USE_CUDA
    if (nn->backend == BACKEND_GPU) {
        return nn_forward_cached_gpu(nn, input);
    }
#endif

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

Matrix *nn_forward(Network *nn, Matrix *input)
{
    ForwardCache *cache = nn_forward_cached(nn, input);
    Matrix *result = matrix_copy(cache->output);
    forward_cache_free(cache);
    return result;
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
                if (y <= 0.0f) {
                    continue;
                }
                nn_float y_hat = forward_cache->output
                                     ->data[row * forward_cache->output->cols + col];
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
            backward_cache->loss +=
                -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
        }
        backward_cache->loss /= batch;
    } else {
        for (int col = 0; col < batch; col++) {
            for (int row = 0; row < output_rows; row++) {
                nn_float y = target->data[row * target->cols + col];
                nn_float y_hat = forward_cache->output
                                     ->data[row * forward_cache->output->cols + col];
                y_hat = fmaxf(epsilon, fminf(1.0f - epsilon, y_hat));
                backward_cache->loss +=
                    -(y * logf(y_hat) + (1.0f - y) * logf(1.0f - y_hat));
            }
        }
        backward_cache->loss /= (output_rows * batch);
    }

    backward_cache->output_gradient = matrix_subtract(forward_cache->output, target);

    backward_cache->grad_norm = 0.0f;
    int total =
        backward_cache->output_gradient->rows * backward_cache->output_gradient->cols;
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

TrainStepStats nn_train_batch_cpu(Network *nn, Matrix *input, Matrix *target,
                                  int batch_size, Matrix *output_buffer)
{
    TrainStepStats stats = {.loss = 0.0f,
                            .grad_norm = 0.0f,
                            .samples = batch_size,
                            .mfu = 0.0f,
                            .vram_used_bytes = 0,
                            .vram_total_bytes = 0};

    ForwardCache *forward_cache = nn_forward_cached(nn, input);

    if (output_buffer != NULL && output_buffer->rows == forward_cache->output->rows &&
        output_buffer->cols >= batch_size) {
        for (int row = 0; row < forward_cache->output->rows; row++) {
            for (int col = 0; col < batch_size; col++) {
                output_buffer->data[row * output_buffer->cols + col] =
                    forward_cache->output
                        ->data[row * forward_cache->output->cols + col];
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
