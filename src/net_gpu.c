#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "net_internal.h"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gpu_backend.h"
#include "matrix.h"

#define CUDA_CALL(expr)                                                                \
    do {                                                                               \
        cudaError_t _cuda_status = (expr);                                             \
        if (_cuda_status != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                    cudaGetErrorString(_cuda_status));                                 \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

typedef struct GpuWorkspace {
    nn_float *d_input;
    nn_float *d_target;
    nn_float *d_grad;
    nn_float *d_temp;
    nn_float *h_temp;
    size_t input_capacity;
    size_t target_capacity;
    size_t grad_capacity;
    size_t temp_capacity;
    size_t h_temp_capacity;
    unsigned long long seed;
    unsigned long long step;
    double theoretical_tflops;
    size_t total_vram_bytes;
    int total_vram_known;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    int events_initialized;
} GpuWorkspace;

static void gpu_workspace_ensure(GpuWorkspace *workspace, size_t input_elems,
                                 size_t target_elems, size_t grad_elems);
static void gpu_workspace_ensure_temp(GpuWorkspace *workspace, size_t elements);
static void dropout_ensure_device_mask(DropoutLayerData *data, int elements);
static void skip_connection_ensure_saved_device(SkipConnectionData *data,
                                                size_t elements);
static void skip_connection_ensure_grad_device(SkipConnectionData *data,
                                               size_t elements);

GpuWorkspace *gpu_workspace_create(void)
{
    GpuWorkspace *workspace = (GpuWorkspace *) calloc(1, sizeof(GpuWorkspace));
    if (workspace == NULL) {
        return NULL;
    }

    workspace->theoretical_tflops = 209.5;
    workspace->total_vram_bytes = 0;
    workspace->total_vram_known = 0;
    const char *env = getenv("NNC_GPU_TFLOPS");
    if (env != NULL) {
        char *endptr = NULL;
        double value = strtod(env, &endptr);
        if (endptr != env && value > 0.0) {
            workspace->theoretical_tflops = value;
        }
    }

    if (cudaEventCreate(&workspace->start_event) == cudaSuccess &&
        cudaEventCreate(&workspace->stop_event) == cudaSuccess) {
        workspace->events_initialized = 1;
    } else {
        workspace->events_initialized = 0;
    }

    return workspace;
}

void gpu_workspace_destroy(GpuWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    if (workspace->d_input != NULL) {
        CUDA_CALL(cudaFree(workspace->d_input));
    }
    if (workspace->d_target != NULL) {
        CUDA_CALL(cudaFree(workspace->d_target));
    }
    if (workspace->d_grad != NULL) {
        CUDA_CALL(cudaFree(workspace->d_grad));
    }
    if (workspace->d_temp != NULL) {
        CUDA_CALL(cudaFree(workspace->d_temp));
    }
    if (workspace->h_temp != NULL) {
        free(workspace->h_temp);
    }
    if (workspace->events_initialized) {
        cudaEventDestroy(workspace->start_event);
        cudaEventDestroy(workspace->stop_event);
    }
    free(workspace);
}

static void gpu_workspace_ensure(GpuWorkspace *workspace, size_t input_elems,
                                 size_t target_elems, size_t grad_elems)
{
    if (workspace == NULL) {
        return;
    }
    if (input_elems > workspace->input_capacity) {
        if (workspace->d_input != NULL) {
            CUDA_CALL(cudaFree(workspace->d_input));
        }
        CUDA_CALL(
            cudaMalloc((void **) &workspace->d_input, input_elems * sizeof(nn_float)));
        workspace->input_capacity = input_elems;
    }
    if (target_elems > workspace->target_capacity) {
        if (workspace->d_target != NULL) {
            CUDA_CALL(cudaFree(workspace->d_target));
        }
        CUDA_CALL(cudaMalloc((void **) &workspace->d_target,
                             target_elems * sizeof(nn_float)));
        workspace->target_capacity = target_elems;
    }
    if (grad_elems > workspace->grad_capacity) {
        if (workspace->d_grad != NULL) {
            CUDA_CALL(cudaFree(workspace->d_grad));
        }
        CUDA_CALL(
            cudaMalloc((void **) &workspace->d_grad, grad_elems * sizeof(nn_float)));
        workspace->grad_capacity = grad_elems;
    }
    size_t temp_required = input_elems;
    if (target_elems > temp_required) {
        temp_required = target_elems;
    }
    if (grad_elems > temp_required) {
        temp_required = grad_elems;
    }
    if (temp_required > workspace->temp_capacity) {
        if (workspace->d_temp != NULL) {
            CUDA_CALL(cudaFree(workspace->d_temp));
        }
        if (temp_required > 0) {
            CUDA_CALL(cudaMalloc((void **) &workspace->d_temp,
                                 temp_required * sizeof(nn_float)));
        } else {
            workspace->d_temp = NULL;
        }
        workspace->temp_capacity = temp_required;
    }
    if (temp_required > workspace->h_temp_capacity) {
        nn_float *new_host = NULL;
        if (temp_required > 0) {
            new_host = (nn_float *) malloc(temp_required * sizeof(nn_float));
            if (new_host == NULL) {
                fprintf(stderr, "Failed to allocate host workspace buffer\n");
                exit(EXIT_FAILURE);
            }
        }
        free(workspace->h_temp);
        workspace->h_temp = new_host;
        workspace->h_temp_capacity = temp_required;
    }
}

static void gpu_workspace_ensure_temp(GpuWorkspace *workspace, size_t elements)
{
    if (workspace == NULL) {
        return;
    }
    if (elements > workspace->temp_capacity) {
        if (workspace->d_temp != NULL) {
            CUDA_CALL(cudaFree(workspace->d_temp));
        }
        CUDA_CALL(cudaMalloc((void **) &workspace->d_temp,
                             elements * sizeof(nn_float)));
        workspace->temp_capacity = elements;
    }
    if (elements > workspace->h_temp_capacity) {
        nn_float *new_host = (nn_float *) malloc(elements * sizeof(nn_float));
        if (new_host == NULL) {
            fprintf(stderr, "Failed to allocate host temp buffer\n");
            exit(EXIT_FAILURE);
        }
        free(workspace->h_temp);
        workspace->h_temp = new_host;
        workspace->h_temp_capacity = elements;
    }
}

static void dropout_ensure_device_mask(DropoutLayerData *data, int elements)
{
    if (data == NULL) {
        return;
    }
    size_t required = (size_t) elements;
    if (required > data->mask_capacity) {
        if (data->d_mask != NULL) {
            CUDA_CALL(cudaFree(data->d_mask));
        }
        CUDA_CALL(cudaMalloc((void **) &data->d_mask, required * sizeof(nn_float)));
        data->mask_capacity = required;
    }
}

static void skip_connection_ensure_saved_device(SkipConnectionData *data,
                                                size_t elements)
{
    if (data == NULL) {
        return;
    }
    if (elements > data->saved_capacity) {
        if (data->d_saved_input != NULL) {
            CUDA_CALL(cudaFree(data->d_saved_input));
        }
        CUDA_CALL(cudaMalloc((void **) &data->d_saved_input, elements * sizeof(nn_float)));
        data->saved_capacity = elements;
    }
}

static void skip_connection_ensure_grad_device(SkipConnectionData *data,
                                               size_t elements)
{
    if (data == NULL || elements == 0) {
        return;
    }
    if (elements > data->grad_capacity) {
        if (data->d_grad_accum != NULL) {
            CUDA_CALL(cudaFree(data->d_grad_accum));
        }
        CUDA_CALL(
            cudaMalloc((void **) &data->d_grad_accum, elements * sizeof(nn_float)));
        data->grad_capacity = elements;
    }
}

ForwardCache *nn_forward_cached_gpu(Network *nn, Matrix *input)
{
    ForwardCache *cache = (ForwardCache *) malloc(sizeof(ForwardCache));
    if (cache == NULL) {
        return NULL;
    }

    int batch_size = input->cols;
    size_t input_elems = (size_t) input->rows * (size_t) batch_size;
    gpu_workspace_ensure(nn->gpu_workspace, input_elems, 0, 0);
    if (input_elems > 0) {
        copy_matrix_to_column_major(input, nn->gpu_workspace->h_temp);
        CUDA_CALL(cudaMemcpy(nn->gpu_workspace->d_input, nn->gpu_workspace->h_temp,
                             input_elems * sizeof(nn_float), cudaMemcpyHostToDevice));
    }

    nn_float *d_current = nn->gpu_workspace->d_input;
    int current_rows = input->rows;

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            d_current = dense_gpu_forward_device(data->gpu, d_current, batch_size);
            current_rows = data->weights->rows;
            break;
        }
        case LAYER_ACTIVATION: {
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            activation_gpu_forward(data->kind, d_current, current_rows, batch_size);
            data->d_last_output = d_current;
            data->last_rows = current_rows;
            data->last_cols = batch_size;
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            if (data->training) {
                int elements = current_rows * batch_size;
                dropout_ensure_device_mask(data, elements);
                unsigned long long seed =
                    nn->gpu_workspace->seed + nn->gpu_workspace->step++;
                dropout_gpu_forward(d_current, data->d_mask, elements, data->rate,
                                    seed);
            }
            break;
        }
        case LAYER_SOFTMAX: {
            SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;
            softmax_gpu_forward(d_current, current_rows, batch_size);
            data->d_last_output = d_current;
            data->last_rows = current_rows;
            data->last_cols = batch_size;
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            d_current = conv2d_gpu_forward_device(data->gpu, d_current, batch_size);
            current_rows =
                data->out_channels * data->output_height * data->output_width;
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            batchnorm_layer_gpu_ensure_capacity(data, batch_size);
            batchnorm_layer_gpu_copy_params_to_device(data);
            batchnorm_layer_gpu_copy_running_to_device(data);
            batchnorm_gpu_forward(d_current, d_current, data->d_gamma, data->d_beta,
                                  data->d_running_mean, data->d_running_var,
                                  data->d_batch_mean, data->d_batch_var,
                                  data->d_normalized, data->channels, data->spatial,
                                  batch_size, data->training, data->momentum,
                                  data->epsilon);
            if (data->training) {
                batchnorm_layer_gpu_copy_running_to_host(data);
            }
            break;
        }
        default:
            break;
        }
    }

    Matrix *output = matrix_create(current_rows, batch_size);
    if ((size_t) current_rows * batch_size > 0) {
        CUDA_CALL(cudaMemcpy(nn->gpu_workspace->h_temp, d_current,
                             (size_t) current_rows * batch_size * sizeof(nn_float),
                             cudaMemcpyDeviceToHost));
        copy_column_major_to_matrix(output, nn->gpu_workspace->h_temp);
    }
    cache->output = output;
    return cache;
}

Matrix *nn_forward_gpu(Network *nn, Matrix *input)
{
    ForwardCache *cache = nn_forward_cached_gpu(nn, input);
    if (cache == NULL) {
        return NULL;
    }
    Matrix *result = matrix_copy(cache->output);
    forward_cache_free(cache);
    return result;
}

TrainStepStats nn_train_batch_gpu(Network *nn, Matrix *input, Matrix *target,
                                  int batch_size, Matrix *output_buffer)
{
    TrainStepStats stats = {.loss = 0.0f,
                            .grad_norm = 0.0f,
                            .samples = batch_size,
                            .mfu = 0.0f,
                            .vram_used_bytes = 0,
                            .vram_total_bytes = 0};
    if (batch_size <= 0) {
        stats.samples = 0;
        return stats;
    }

    if (nn->flops_dirty) {
        nn->flops_per_sample = compute_network_flops_per_sample(nn);
        nn->flops_dirty = 0;
    }
    double flops_total = nn->flops_per_sample * (double) batch_size;

    GpuWorkspace *workspace = nn->gpu_workspace;
    int measure_metrics = workspace != NULL && workspace->events_initialized && nn->log_memory;
    cudaEvent_t start_event = measure_metrics ? workspace->start_event : NULL;
    cudaEvent_t stop_event = measure_metrics ? workspace->stop_event : NULL;
    if (start_event != NULL) {
        CUDA_CALL(cudaEventRecord(start_event, 0));
    }
    size_t input_elems = (size_t) input->rows * (size_t) batch_size;
    size_t target_elems = (size_t) target->rows * (size_t) batch_size;
    gpu_workspace_ensure(workspace, input_elems, target_elems, target_elems);

    if (input_elems > 0) {
        copy_matrix_to_column_major(input, workspace->h_temp);
        CUDA_CALL(cudaMemcpy(workspace->d_input, workspace->h_temp,
                             input_elems * sizeof(nn_float), cudaMemcpyHostToDevice));
    }
    if (target_elems > 0) {
        copy_matrix_to_column_major(target, workspace->h_temp);
        CUDA_CALL(cudaMemcpy(workspace->d_target, workspace->h_temp,
                             target_elems * sizeof(nn_float), cudaMemcpyHostToDevice));
    }

    nn_float *d_current = workspace->d_input;
    int current_rows = input->rows;

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            d_current = dense_gpu_forward_device(data->gpu, d_current, batch_size);
            current_rows = data->weights->rows;
            break;
        }
        case LAYER_ACTIVATION: {
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            activation_gpu_forward(data->kind, d_current, current_rows, batch_size);
            data->d_last_output = d_current;
            data->last_rows = current_rows;
            data->last_cols = batch_size;
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            if (data->training) {
                int elements = current_rows * batch_size;
                dropout_ensure_device_mask(data, elements);
                unsigned long long seed = workspace->seed + workspace->step++;
                dropout_gpu_forward(d_current, data->d_mask, elements, data->rate,
                                    seed);
            }
            break;
        }
        case LAYER_SOFTMAX: {
            SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;
            softmax_gpu_forward(d_current, current_rows, batch_size);
            data->d_last_output = d_current;
            data->last_rows = current_rows;
            data->last_cols = batch_size;
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            d_current = conv2d_gpu_forward_device(data->gpu, d_current, batch_size);
            current_rows =
                data->out_channels * data->output_height * data->output_width;
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            batchnorm_layer_gpu_ensure_capacity(data, batch_size);
            batchnorm_layer_gpu_copy_params_to_device(data);
            batchnorm_layer_gpu_copy_running_to_device(data);
            batchnorm_gpu_forward(d_current, d_current, data->d_gamma, data->d_beta,
                                  data->d_running_mean, data->d_running_var,
                                  data->d_batch_mean, data->d_batch_var,
                                  data->d_normalized, data->channels, data->spatial,
                                  batch_size, data->training, data->momentum,
                                  data->epsilon);
            if (data->training) {
                batchnorm_layer_gpu_copy_running_to_host(data);
            }
            break;
        }
        default:
            break;
        }
    }

    Matrix *host_output = output_buffer;
    Matrix *temp_output = NULL;
    if (host_output == NULL) {
        temp_output = matrix_create(current_rows, batch_size);
        host_output = temp_output;
    }
    size_t output_elements = (size_t) current_rows * (size_t) batch_size;
    if (output_elements > 0) {
        CUDA_CALL(cudaMemcpy(workspace->h_temp, d_current,
                             output_elements * sizeof(nn_float),
                             cudaMemcpyDeviceToHost));
        copy_column_major_to_matrix(host_output, workspace->h_temp);
    }
    host_output->rows = current_rows;
    host_output->cols = batch_size;

    stats.loss = compute_loss_from_output(nn, host_output, target, batch_size);

    if (output_elements > 0) {
        CUDA_CALL(cudaMemcpy(workspace->d_grad, d_current,
                             output_elements * sizeof(nn_float),
                             cudaMemcpyDeviceToDevice));
    }
    vector_subtract_inplace(workspace->d_grad, workspace->d_target,
                            current_rows * batch_size);

    stats.grad_norm = gpu_vector_l2_norm(workspace->d_grad, current_rows * batch_size) /
                      (nn_float) batch_size;

    nn_float *d_grad = workspace->d_grad;
    int grad_rows = current_rows;

    for (size_t idx = nn->layer_count; idx-- > 0;) {
        Layer *layer = nn->layers[idx];
        switch (layer->type) {
        case LAYER_SOFTMAX: {
            SoftmaxLayerData *data = (SoftmaxLayerData *) layer->data;
            size_t elements = (size_t) data->last_rows * (size_t) batch_size;
            gpu_workspace_ensure_temp(workspace, elements);
            CUDA_CALL(cudaMemcpy(workspace->d_temp, d_grad, elements * sizeof(nn_float),
                                 cudaMemcpyDeviceToDevice));
            softmax_gpu_backward(data->d_last_output, workspace->d_temp,
                                 workspace->d_grad, data->last_rows, batch_size);
            d_grad = workspace->d_grad;
            grad_rows = data->last_rows;
            break;
        }
        case LAYER_SKIP_ADD: {
            SkipConnectionData *data = (SkipConnectionData *) layer->data;
            size_t elements = (size_t) grad_rows * (size_t) batch_size;
            if (elements > 0) {
                skip_connection_ensure_grad_device(data, elements);
                CUDA_CALL(cudaMemcpy(data->d_grad_accum, d_grad,
                                     elements * sizeof(nn_float),
                                     cudaMemcpyDeviceToDevice));
            }
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            batchnorm_layer_gpu_ensure_capacity(data, batch_size);
            nn_float *next_grad = data->d_grad_input;
            batchnorm_gpu_backward(d_grad, next_grad, data->d_gamma, data->d_batch_mean,
                                   data->d_batch_var, data->d_normalized, data->d_grad_gamma,
                                   data->d_grad_beta, data->channels, data->spatial, batch_size,
                                   data->epsilon);
            batchnorm_layer_gpu_copy_grads_to_host(data);
            d_grad = next_grad;
            grad_rows = data->channels * data->spatial;
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            if (data->training && data->d_mask != NULL) {
                int elements = grad_rows * batch_size;
                dropout_gpu_backward(d_grad, data->d_mask, elements);
            }
            break;
        }
        case LAYER_ACTIVATION: {
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            activation_gpu_backward(data->kind, data->d_last_output, d_grad,
                                    data->last_rows, batch_size);
            grad_rows = data->last_rows;
            break;
        }
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            d_grad = dense_gpu_backward_device(data->gpu, d_grad, batch_size);
            grad_rows = data->weights->cols;
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            d_grad = conv2d_gpu_backward_device(data->gpu, d_grad, batch_size);
            grad_rows = data->in_channels * data->input_height * data->input_width;
            break;
        }
        default:
            break;
        }
    }

    float elapsed_ms = 0.0f;
    if (stop_event != NULL) {
        CUDA_CALL(cudaEventRecord(stop_event, 0));
        CUDA_CALL(cudaEventSynchronize(stop_event));
        CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    }

    if (nn->log_memory) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (total_bytes > 0) {
            stats.vram_total_bytes = total_bytes;
            stats.vram_used_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
            workspace->total_vram_bytes = total_bytes;
            workspace->total_vram_known = 1;
        } else if (workspace->total_vram_known && workspace->total_vram_bytes > 0) {
            stats.vram_total_bytes = workspace->total_vram_bytes;
            stats.vram_used_bytes = 0;
        }
    }

    if (measure_metrics) {
        double elapsed_seconds = elapsed_ms / 1000.0;
        double achieved_tflops = 0.0;
        if (elapsed_seconds > 0.0 && flops_total > 0.0) {
            achieved_tflops = (flops_total / elapsed_seconds) / 1e12;
        }
        double theoretical = workspace->theoretical_tflops;
        if (theoretical <= 0.0) {
            theoretical = 209.5;
        }
        double mfu_ratio = (theoretical > 0.0) ? (achieved_tflops / theoretical) : 0.0;
        if (mfu_ratio < 0.0) {
            mfu_ratio = 0.0;
        }
        stats.mfu = (nn_float) mfu_ratio;
    } else {
        stats.mfu = 0.0f;
    }

    if (temp_output != NULL) {
        matrix_free(temp_output);
    }

    return stats;
}

void gpu_workspace_set_seed(GpuWorkspace *workspace, unsigned long long seed)
{
    if (workspace == NULL) {
        return;
    }
    workspace->seed = seed;
    workspace->step = 0;
}

void gpu_workspace_set_theoretical_tflops(GpuWorkspace *workspace, double tflops)
{
    if (workspace == NULL || tflops <= 0.0) {
        return;
    }
    workspace->theoretical_tflops = tflops;
}

#endif /* USE_CUDA */
