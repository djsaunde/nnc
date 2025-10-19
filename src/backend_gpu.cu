#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_backend.h"
#include "net.h"

#ifdef __cplusplus
extern "C" {
#endif

static void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void check_cublas(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", msg, status);
        exit(EXIT_FAILURE);
    }
}

static cublasHandle_t global_handle(void)
{
    static cublasHandle_t handle = NULL;
    if (handle == NULL) {
        check_cublas(cublasCreate(&handle), "cublasCreate");
    }
    return handle;
}

typedef struct DenseLayerGpuContext {
    int input_size;
    int output_size;
    int capacity;
    nn_float *d_weights;
    nn_float *d_bias;
    nn_float *d_last_input;
    nn_float *d_output;
    nn_float *d_grad_input;
    nn_float *d_grad_weights;
    nn_float *d_grad_bias;
    nn_float *d_m_weights;
    nn_float *d_v_weights;
    nn_float *d_m_bias;
    nn_float *d_v_bias;
    int adam_initialized;
} DenseLayerGpuContext;

typedef struct Conv2DLayerGpuContext {
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
    int input_size;
    int output_size;
    int weight_elements;
    int capacity;
    nn_float *d_weights;
    nn_float *d_bias;
    nn_float *d_last_input;
    nn_float *d_output;
    nn_float *d_grad_input;
    nn_float *d_grad_weights;
    nn_float *d_grad_bias;
    nn_float *d_m_weights;
    nn_float *d_v_weights;
    nn_float *d_m_bias;
    nn_float *d_v_bias;
    int adam_initialized;
} Conv2DLayerGpuContext;

static void ensure_capacity(DenseLayerGpuContext *ctx, int batch)
{
    if (batch <= ctx->capacity) {
        return;
    }

    size_t input_bytes = (size_t) ctx->input_size * (size_t) batch * sizeof(nn_float);
    size_t output_bytes = (size_t) ctx->output_size * (size_t) batch * sizeof(nn_float);

    if (ctx->d_last_input != NULL) {
        check_cuda(cudaFree(ctx->d_last_input), "cudaFree last_input");
    }
    if (ctx->d_output != NULL) {
        check_cuda(cudaFree(ctx->d_output), "cudaFree output");
    }
    if (ctx->d_grad_input != NULL) {
        check_cuda(cudaFree(ctx->d_grad_input), "cudaFree grad_input");
    }

    check_cuda(cudaMalloc((void **) &ctx->d_last_input, input_bytes), "cudaMalloc last_input");
    check_cuda(cudaMalloc((void **) &ctx->d_output, output_bytes), "cudaMalloc output");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_input, input_bytes), "cudaMalloc grad_input");

    ctx->capacity = batch;
}

static void dense_gpu_ensure_adam_state(DenseLayerGpuContext *ctx)
{
    size_t weight_bytes = (size_t) ctx->output_size * (size_t) ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);
    if (ctx->d_m_weights == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_m_weights, weight_bytes), "cudaMalloc m_weights");
    }
    if (ctx->d_v_weights == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_v_weights, weight_bytes), "cudaMalloc v_weights");
    }
    if (ctx->d_m_bias == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_m_bias, bias_bytes), "cudaMalloc m_bias");
    }
    if (ctx->d_v_bias == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_v_bias, bias_bytes), "cudaMalloc v_bias");
    }
    if (!ctx->adam_initialized) {
        check_cuda(cudaMemset(ctx->d_m_weights, 0, weight_bytes), "cudaMemset m_weights");
        check_cuda(cudaMemset(ctx->d_v_weights, 0, weight_bytes), "cudaMemset v_weights");
        check_cuda(cudaMemset(ctx->d_m_bias, 0, bias_bytes), "cudaMemset m_bias");
        check_cuda(cudaMemset(ctx->d_v_bias, 0, bias_bytes), "cudaMemset v_bias");
        ctx->adam_initialized = 1;
    }
}

static void conv2d_ensure_capacity(Conv2DLayerGpuContext *ctx, int batch)
{
    if (batch <= ctx->capacity) {
        return;
    }

    size_t input_bytes = (size_t) ctx->input_size * (size_t) batch * sizeof(nn_float);
    size_t output_bytes = (size_t) ctx->output_size * (size_t) batch * sizeof(nn_float);

    if (ctx->d_last_input != NULL) {
        check_cuda(cudaFree(ctx->d_last_input), "cudaFree conv last_input");
    }
    if (ctx->d_output != NULL) {
        check_cuda(cudaFree(ctx->d_output), "cudaFree conv output");
    }
    if (ctx->d_grad_input != NULL) {
        check_cuda(cudaFree(ctx->d_grad_input), "cudaFree conv grad_input");
    }

    check_cuda(cudaMalloc((void **) &ctx->d_last_input, input_bytes), "cudaMalloc conv last_input");
    check_cuda(cudaMalloc((void **) &ctx->d_output, output_bytes), "cudaMalloc conv output");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_input, input_bytes),
               "cudaMalloc conv grad_input");

    ctx->capacity = batch;
}

static void conv2d_gpu_ensure_adam_state(Conv2DLayerGpuContext *ctx)
{
    size_t weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);
    if (ctx->d_m_weights == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_m_weights, weight_bytes),
                   "cudaMalloc conv m_weights");
    }
    if (ctx->d_v_weights == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_v_weights, weight_bytes),
                   "cudaMalloc conv v_weights");
    }
    if (ctx->d_m_bias == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_m_bias, bias_bytes),
                   "cudaMalloc conv m_bias");
    }
    if (ctx->d_v_bias == NULL) {
        check_cuda(cudaMalloc((void **) &ctx->d_v_bias, bias_bytes),
                   "cudaMalloc conv v_bias");
    }
    if (!ctx->adam_initialized) {
        if (ctx->d_m_weights != NULL) {
            check_cuda(cudaMemset(ctx->d_m_weights, 0, weight_bytes),
                       "cudaMemset conv m_weights");
        }
        if (ctx->d_v_weights != NULL) {
            check_cuda(cudaMemset(ctx->d_v_weights, 0, weight_bytes),
                       "cudaMemset conv v_weights");
        }
        if (ctx->d_m_bias != NULL) {
            check_cuda(cudaMemset(ctx->d_m_bias, 0, bias_bytes), "cudaMemset conv m_bias");
        }
        if (ctx->d_v_bias != NULL) {
            check_cuda(cudaMemset(ctx->d_v_bias, 0, bias_bytes), "cudaMemset conv v_bias");
        }
        ctx->adam_initialized = 1;
    }
}

DenseLayerGpuContext *dense_gpu_create(int input_size, int output_size)
{
    DenseLayerGpuContext *ctx = (DenseLayerGpuContext *) calloc(1, sizeof(DenseLayerGpuContext));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->input_size = input_size;
    ctx->output_size = output_size;
    ctx->capacity = 0;

    size_t weight_bytes = (size_t) output_size * (size_t) input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) output_size * sizeof(nn_float);

    check_cuda(cudaMalloc((void **) &ctx->d_weights, weight_bytes), "cudaMalloc d_weights");
    check_cuda(cudaMalloc((void **) &ctx->d_bias, bias_bytes), "cudaMalloc d_bias");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_weights, weight_bytes), "cudaMalloc d_grad_weights");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_bias, bias_bytes), "cudaMalloc d_grad_bias");
    ctx->d_m_weights = NULL;
    ctx->d_v_weights = NULL;
    ctx->d_m_bias = NULL;
    ctx->d_v_bias = NULL;
    ctx->adam_initialized = 0;

    return ctx;
}

void dense_gpu_destroy(DenseLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    if (ctx->d_weights != NULL) {
        check_cuda(cudaFree(ctx->d_weights), "cudaFree weights");
    }
    if (ctx->d_bias != NULL) {
        check_cuda(cudaFree(ctx->d_bias), "cudaFree bias");
    }
    if (ctx->d_grad_weights != NULL) {
        check_cuda(cudaFree(ctx->d_grad_weights), "cudaFree grad_weights");
    }
    if (ctx->d_grad_bias != NULL) {
        check_cuda(cudaFree(ctx->d_grad_bias), "cudaFree grad_bias");
    }
    if (ctx->d_last_input != NULL) {
        check_cuda(cudaFree(ctx->d_last_input), "cudaFree last_input");
    }
    if (ctx->d_output != NULL) {
        check_cuda(cudaFree(ctx->d_output), "cudaFree output");
    }
    if (ctx->d_grad_input != NULL) {
        check_cuda(cudaFree(ctx->d_grad_input), "cudaFree grad_input");
    }
    if (ctx->d_m_weights != NULL) {
        check_cuda(cudaFree(ctx->d_m_weights), "cudaFree m_weights");
    }
    if (ctx->d_v_weights != NULL) {
        check_cuda(cudaFree(ctx->d_v_weights), "cudaFree v_weights");
    }
    if (ctx->d_m_bias != NULL) {
        check_cuda(cudaFree(ctx->d_m_bias), "cudaFree m_bias");
    }
    if (ctx->d_v_bias != NULL) {
        check_cuda(cudaFree(ctx->d_v_bias), "cudaFree v_bias");
    }
    free(ctx);
}

static void row_to_col(nn_float *dst, const nn_float *src, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

static void col_to_row(nn_float *dst, const nn_float *src, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
}

void dense_gpu_set_weights(DenseLayerGpuContext *ctx, const nn_float *weights, const nn_float *bias)
{
    size_t weight_bytes = (size_t) ctx->output_size * (size_t) ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);

    nn_float *temp = (nn_float *) malloc(weight_bytes);
    if (temp == NULL) {
        fprintf(stderr, "Failed to allocate temporary weight buffer\n");
        exit(EXIT_FAILURE);
    }
    row_to_col(temp, weights, ctx->output_size, ctx->input_size);

    check_cuda(cudaMemcpy(ctx->d_weights, temp, weight_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy d_weights");
    free(temp);

    check_cuda(cudaMemcpy(ctx->d_bias, bias, bias_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy d_bias");
}

void dense_gpu_get_weights(DenseLayerGpuContext *ctx, nn_float *weights, nn_float *bias)
{
    size_t weight_bytes = (size_t) ctx->output_size * (size_t) ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);

    nn_float *temp = (nn_float *) malloc(weight_bytes);
    if (temp == NULL) {
        fprintf(stderr, "Failed to allocate temporary weight buffer\n");
        exit(EXIT_FAILURE);
    }
    check_cuda(cudaMemcpy(temp, ctx->d_weights, weight_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy weights host");
    col_to_row(weights, temp, ctx->output_size, ctx->input_size);
    free(temp);

    check_cuda(cudaMemcpy(bias, ctx->d_bias, bias_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy bias host");
}

static __global__ void bias_add_kernel(nn_float *output, const nn_float *bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }
    int row = idx % rows;
    output[idx] += bias[row];
}

static __global__ void bias_grad_kernel(const nn_float *grad_output, nn_float *grad_bias, int rows,
                                        int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    nn_float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += grad_output[row + col * rows];
    }
    atomicAdd(&grad_bias[row], sum);
}

static __global__ void adamw_update_kernel(nn_float *weights, nn_float *grads, nn_float *m,
                                           nn_float *v, int elements, nn_float beta1,
                                           nn_float beta2, nn_float one_minus_beta1,
                                           nn_float one_minus_beta2, nn_float epsilon,
                                           nn_float learning_rate, nn_float weight_decay,
                                           nn_float inv_bias_correction1,
                                           nn_float inv_bias_correction2, nn_float grad_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    nn_float grad = grads[idx] * grad_scale;
    nn_float m_val = m[idx] = beta1 * m[idx] + one_minus_beta1 * grad;
    nn_float v_val = v[idx] = beta2 * v[idx] + one_minus_beta2 * grad * grad;
    nn_float m_hat = m_val * inv_bias_correction1;
    nn_float v_hat = v_val * inv_bias_correction2;
    nn_float denom = sqrtf(v_hat) + epsilon;
    nn_float update = m_hat / denom;
    nn_float decay = weight_decay > 0.0f ? weight_decay * weights[idx] : 0.0f;
    weights[idx] -= learning_rate * (update + decay);
    grads[idx] = 0.0f;
}

nn_float *dense_gpu_forward_device(DenseLayerGpuContext *ctx, const nn_float *d_input, int batch)
{
    if (batch <= 0) {
        return ctx->d_output;
    }
    ensure_capacity(ctx, batch);

    size_t input_bytes = (size_t) ctx->input_size * (size_t) batch * sizeof(nn_float);
    check_cuda(cudaMemcpy(ctx->d_last_input, d_input, input_bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy last_input");

    const nn_float alpha = 1.0f;
    const nn_float beta = 0.0f;
    cublasHandle_t handle = global_handle();

    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ctx->output_size, batch,
                              ctx->input_size, &alpha, ctx->d_weights, ctx->output_size,
                              ctx->d_last_input, ctx->input_size, &beta, ctx->d_output,
                              ctx->output_size),
                 "cublasSgemm forward");

    int total = ctx->output_size * batch;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 0) {
        bias_add_kernel<<<grid, block>>>(ctx->d_output, ctx->d_bias, ctx->output_size, batch);
        check_cuda(cudaGetLastError(), "bias_add_kernel");
    }

    return ctx->d_output;
}

nn_float *dense_gpu_backward_device(DenseLayerGpuContext *ctx, const nn_float *d_grad_output,
                                    int batch)
{
    if (batch <= 0) {
        return ctx->d_grad_input;
    }
    ensure_capacity(ctx, batch);

    const nn_float alpha = 1.0f;
    const nn_float beta_one = 1.0f;
    const nn_float beta_zero = 0.0f;
    cublasHandle_t handle = global_handle();

    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ctx->output_size, ctx->input_size,
                              batch, &alpha, d_grad_output, ctx->output_size,
                              ctx->d_last_input, ctx->input_size, &beta_one, ctx->d_grad_weights,
                              ctx->output_size),
                 "cublasSgemm grad_weights");

    int rows = ctx->output_size;
    int block = 128;
    int grid = (rows + block - 1) / block;
    bias_grad_kernel<<<grid, block>>>(d_grad_output, ctx->d_grad_bias, ctx->output_size, batch);
    check_cuda(cudaGetLastError(), "bias_grad_kernel");

    check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ctx->input_size, batch,
                              ctx->output_size, &alpha, ctx->d_weights, ctx->output_size,
                              d_grad_output, ctx->output_size, &beta_zero, ctx->d_grad_input,
                              ctx->input_size),
                 "cublasSgemm grad_input");

    return ctx->d_grad_input;
}

void dense_gpu_zero_gradients(DenseLayerGpuContext *ctx)
{
    size_t weight_bytes = (size_t) ctx->output_size * (size_t) ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);
    check_cuda(cudaMemset(ctx->d_grad_weights, 0, weight_bytes), "cudaMemset grad_weights");
    check_cuda(cudaMemset(ctx->d_grad_bias, 0, bias_bytes), "cudaMemset grad_bias");
}

void dense_gpu_reset_adam(DenseLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    size_t weight_bytes = (size_t) ctx->output_size * (size_t) ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);
    if (ctx->d_m_weights != NULL) {
        check_cuda(cudaMemset(ctx->d_m_weights, 0, weight_bytes), "cudaMemset m_weights");
    }
    if (ctx->d_v_weights != NULL) {
        check_cuda(cudaMemset(ctx->d_v_weights, 0, weight_bytes), "cudaMemset v_weights");
    }
    if (ctx->d_m_bias != NULL) {
        check_cuda(cudaMemset(ctx->d_m_bias, 0, bias_bytes), "cudaMemset m_bias");
    }
    if (ctx->d_v_bias != NULL) {
        check_cuda(cudaMemset(ctx->d_v_bias, 0, bias_bytes), "cudaMemset v_bias");
    }
    ctx->adam_initialized = 0;
}

Conv2DLayerGpuContext *conv2d_gpu_create(int in_channels, int out_channels, int input_height,
                                        int input_width, int kernel_h, int kernel_w, int stride_h,
                                        int stride_w, int pad_h, int pad_w)
{
    Conv2DLayerGpuContext *ctx =
        (Conv2DLayerGpuContext *) calloc(1, sizeof(Conv2DLayerGpuContext));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->in_channels = in_channels;
    ctx->out_channels = out_channels;
    ctx->input_height = input_height;
    ctx->input_width = input_width;
    ctx->kernel_h = kernel_h;
    ctx->kernel_w = kernel_w;
    ctx->stride_h = stride_h;
    ctx->stride_w = stride_w;
    ctx->pad_h = pad_h;
    ctx->pad_w = pad_w;
    ctx->output_height = (input_height + 2 * pad_h - kernel_h) / stride_h + 1;
    ctx->output_width = (input_width + 2 * pad_w - kernel_w) / stride_w + 1;
    ctx->input_size = in_channels * input_height * input_width;
    ctx->output_size = out_channels * ctx->output_height * ctx->output_width;
    ctx->weight_elements = out_channels * in_channels * kernel_h * kernel_w;
    ctx->capacity = 0;
    ctx->d_last_input = NULL;
    ctx->d_output = NULL;
    ctx->d_grad_input = NULL;
    ctx->d_m_weights = NULL;
    ctx->d_v_weights = NULL;
    ctx->d_m_bias = NULL;
    ctx->d_v_bias = NULL;
    ctx->adam_initialized = 0;

    size_t weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);

    check_cuda(cudaMalloc((void **) &ctx->d_weights, weight_bytes), "cudaMalloc conv weights");
    check_cuda(cudaMalloc((void **) &ctx->d_bias, bias_bytes), "cudaMalloc conv bias");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_weights, weight_bytes),
               "cudaMalloc conv grad_weights");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_bias, bias_bytes),
               "cudaMalloc conv grad_bias");

    check_cuda(cudaMemset(ctx->d_grad_weights, 0, weight_bytes), "cudaMemset conv grad_weights");
    check_cuda(cudaMemset(ctx->d_grad_bias, 0, bias_bytes), "cudaMemset conv grad_bias");

    return ctx;
}

void conv2d_gpu_destroy(Conv2DLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    if (ctx->d_weights != NULL) {
        check_cuda(cudaFree(ctx->d_weights), "cudaFree conv weights");
    }
    if (ctx->d_bias != NULL) {
        check_cuda(cudaFree(ctx->d_bias), "cudaFree conv bias");
    }
    if (ctx->d_grad_weights != NULL) {
        check_cuda(cudaFree(ctx->d_grad_weights), "cudaFree conv grad_weights");
    }
    if (ctx->d_grad_bias != NULL) {
        check_cuda(cudaFree(ctx->d_grad_bias), "cudaFree conv grad_bias");
    }
    if (ctx->d_last_input != NULL) {
        check_cuda(cudaFree(ctx->d_last_input), "cudaFree conv last_input");
    }
    if (ctx->d_output != NULL) {
        check_cuda(cudaFree(ctx->d_output), "cudaFree conv output");
    }
    if (ctx->d_grad_input != NULL) {
        check_cuda(cudaFree(ctx->d_grad_input), "cudaFree conv grad_input");
    }
    if (ctx->d_m_weights != NULL) {
        check_cuda(cudaFree(ctx->d_m_weights), "cudaFree conv m_weights");
    }
    if (ctx->d_v_weights != NULL) {
        check_cuda(cudaFree(ctx->d_v_weights), "cudaFree conv v_weights");
    }
    if (ctx->d_m_bias != NULL) {
        check_cuda(cudaFree(ctx->d_m_bias), "cudaFree conv m_bias");
    }
    if (ctx->d_v_bias != NULL) {
        check_cuda(cudaFree(ctx->d_v_bias), "cudaFree conv v_bias");
    }
    free(ctx);
}

void conv2d_gpu_set_weights(Conv2DLayerGpuContext *ctx, const nn_float *weights,
                            const nn_float *bias)
{
    size_t weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);
    check_cuda(cudaMemcpy(ctx->d_weights, weights, weight_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy conv weights");
    check_cuda(cudaMemcpy(ctx->d_bias, bias, bias_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy conv bias");
}

static __global__ void conv2d_forward_kernel(const nn_float *input, const nn_float *weights,
                                             const nn_float *bias, nn_float *output, int batch,
                                             int in_channels, int out_channels, int input_height,
                                             int input_width, int kernel_h, int kernel_w,
                                             int stride_h, int stride_w, int pad_h, int pad_w,
                                             int output_height, int output_width, int input_size,
                                             int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * output_size;
    if (idx >= total) {
        return;
    }

    int per_sample = output_size;
    int b = idx / per_sample;
    int rem = idx % per_sample;
    int plane = output_height * output_width;
    int oc = rem / plane;
    int within = rem % plane;
    int oh = within / output_width;
    int ow = within % output_width;

    nn_float sum = bias[oc];
    int kernel_area = kernel_h * kernel_w;
    int weight_cols = in_channels * kernel_area;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= input_height) {
                continue;
            }
            for (int kw = 0; kw < kernel_w; kw++) {
                int iw = ow * stride_w - pad_w + kw;
                if (iw < 0 || iw >= input_width) {
                    continue;
                }
                int input_row = ((ic * input_height + ih) * input_width + iw);
                nn_float input_val = input[input_row + b * input_size];
                int w_idx = oc * weight_cols + ic * kernel_area + kh * kernel_w + kw;
                nn_float weight_val = weights[w_idx];
                sum += weight_val * input_val;
            }
        }
    }

    int output_row = ((oc * output_height + oh) * output_width + ow);
    output[output_row + b * output_size] = sum;
}

nn_float *conv2d_gpu_forward_device(Conv2DLayerGpuContext *ctx, const nn_float *d_input,
                                    int batch)
{
    if (batch <= 0) {
        return ctx->d_output;
    }
    conv2d_ensure_capacity(ctx, batch);

    size_t input_bytes = (size_t) ctx->input_size * (size_t) batch * sizeof(nn_float);
    check_cuda(cudaMemcpy(ctx->d_last_input, d_input, input_bytes, cudaMemcpyDeviceToDevice),
               "cudaMemcpy conv last_input");

    int total = batch * ctx->output_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 0) {
        conv2d_forward_kernel<<<grid, block>>>(ctx->d_last_input, ctx->d_weights, ctx->d_bias,
                                               ctx->d_output, batch, ctx->in_channels,
                                               ctx->out_channels, ctx->input_height,
                                               ctx->input_width, ctx->kernel_h, ctx->kernel_w,
                                               ctx->stride_h, ctx->stride_w, ctx->pad_h, ctx->pad_w,
                                               ctx->output_height, ctx->output_width,
                                               ctx->input_size, ctx->output_size);
        check_cuda(cudaGetLastError(), "conv2d_forward_kernel");
    }

    return ctx->d_output;
}

static __global__ void conv2d_backward_kernel(const nn_float *grad_output,
                                              const nn_float *last_input,
                                              const nn_float *weights, nn_float *grad_input,
                                              nn_float *grad_weights, nn_float *grad_bias,
                                              int batch, int in_channels, int out_channels,
                                              int input_height, int input_width, int kernel_h,
                                              int kernel_w, int stride_h, int stride_w, int pad_h,
                                              int pad_w, int output_height, int output_width,
                                              int input_size, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * output_size;
    if (idx >= total) {
        return;
    }

    int plane = output_height * output_width;
    int kernel_area = kernel_h * kernel_w;
    int weight_cols = in_channels * kernel_area;

    int b = idx / output_size;
    int rem = idx % output_size;
    int oc = rem / plane;
    int within = rem % plane;
    int oh = within / output_width;
    int ow = within % output_width;

    int output_row = ((oc * output_height + oh) * output_width + ow);
    nn_float grad = grad_output[output_row + b * output_size];
    atomicAdd(&grad_bias[oc], grad);

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= input_height) {
                continue;
            }
            for (int kw = 0; kw < kernel_w; kw++) {
                int iw = ow * stride_w - pad_w + kw;
                if (iw < 0 || iw >= input_width) {
                    continue;
                }
                int input_row = ((ic * input_height + ih) * input_width + iw);
                nn_float input_val = last_input[input_row + b * input_size];
                int w_idx = oc * weight_cols + ic * kernel_area + kh * kernel_w + kw;
                atomicAdd(&grad_weights[w_idx], grad * input_val);
                atomicAdd(&grad_input[input_row + b * input_size], grad * weights[w_idx]);
            }
        }
    }
}

nn_float *conv2d_gpu_backward_device(Conv2DLayerGpuContext *ctx, const nn_float *d_grad_output,
                                     int batch)
{
    if (batch <= 0) {
        return ctx->d_grad_input;
    }
    conv2d_ensure_capacity(ctx, batch);

    size_t grad_input_bytes = (size_t) ctx->input_size * (size_t) batch * sizeof(nn_float);
    size_t grad_weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t grad_bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);
    check_cuda(cudaMemset(ctx->d_grad_input, 0, grad_input_bytes),
               "cudaMemset conv grad_input");
    check_cuda(cudaMemset(ctx->d_grad_weights, 0, grad_weight_bytes),
               "cudaMemset conv grad_weights");
    check_cuda(cudaMemset(ctx->d_grad_bias, 0, grad_bias_bytes), "cudaMemset conv grad_bias");

    int total = batch * ctx->output_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 0) {
        conv2d_backward_kernel<<<grid, block>>>(d_grad_output, ctx->d_last_input, ctx->d_weights,
                                                ctx->d_grad_input, ctx->d_grad_weights,
                                                ctx->d_grad_bias, batch, ctx->in_channels,
                                                ctx->out_channels, ctx->input_height,
                                                ctx->input_width, ctx->kernel_h, ctx->kernel_w,
                                                ctx->stride_h, ctx->stride_w, ctx->pad_h,
                                                ctx->pad_w, ctx->output_height, ctx->output_width,
                                                ctx->input_size, ctx->output_size);
        check_cuda(cudaGetLastError(), "conv2d_backward_kernel");
    }

    return ctx->d_grad_input;
}

void conv2d_gpu_zero_gradients(Conv2DLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    size_t grad_weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t grad_bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);
    check_cuda(cudaMemset(ctx->d_grad_weights, 0, grad_weight_bytes),
               "cudaMemset conv grad_weights");
    check_cuda(cudaMemset(ctx->d_grad_bias, 0, grad_bias_bytes), "cudaMemset conv grad_bias");
}

void conv2d_gpu_apply_gradients(Conv2DLayerGpuContext *ctx, nn_float learning_rate,
                                int optimizer_kind, nn_float beta1, nn_float beta2,
                                nn_float epsilon, nn_float weight_decay,
                                nn_float inv_bias_correction1, nn_float inv_bias_correction2,
                                nn_float grad_scale)
{
    if (ctx == NULL) {
        return;
    }

    if ((OptimizerKind) optimizer_kind == OPTIMIZER_ADAMW) {
        conv2d_gpu_ensure_adam_state(ctx);
        nn_float one_minus_beta1 = 1.0f - beta1;
        nn_float one_minus_beta2 = 1.0f - beta2;
        int weight_elements = ctx->weight_elements;
        int bias_elements = ctx->out_channels;
        int block = 256;
        int grid_weights = (weight_elements + block - 1) / block;
        adamw_update_kernel<<<grid_weights, block>>>(ctx->d_weights, ctx->d_grad_weights,
                                                     ctx->d_m_weights, ctx->d_v_weights,
                                                     weight_elements, beta1, beta2,
                                                     one_minus_beta1, one_minus_beta2, epsilon,
                                                     learning_rate, weight_decay,
                                                     inv_bias_correction1, inv_bias_correction2,
                                                     grad_scale);
        check_cuda(cudaGetLastError(), "conv adamw weights");
        int grid_bias = (bias_elements + block - 1) / block;
        adamw_update_kernel<<<grid_bias, block>>>(ctx->d_bias, ctx->d_grad_bias, ctx->d_m_bias,
                                                  ctx->d_v_bias, bias_elements, beta1, beta2,
                                                  one_minus_beta1, one_minus_beta2, epsilon,
                                                  learning_rate, 0.0f, inv_bias_correction1,
                                                  inv_bias_correction2, grad_scale);
        check_cuda(cudaGetLastError(), "conv adamw bias");
        conv2d_gpu_zero_gradients(ctx);
        return;
    }

    cublasHandle_t handle = global_handle();
    if (weight_decay != 0.0f) {
        check_cublas(cublasSaxpy(handle, ctx->weight_elements, &weight_decay, ctx->d_weights, 1,
                                  ctx->d_grad_weights, 1),
                     "conv cublasSaxpy weight decay");
    }
    nn_float alpha = -learning_rate;
    check_cublas(cublasSaxpy(handle, ctx->weight_elements, &alpha, ctx->d_grad_weights, 1,
                              ctx->d_weights, 1),
                 "conv cublasSaxpy weights");
    check_cublas(cublasSaxpy(handle, ctx->out_channels, &alpha, ctx->d_grad_bias, 1, ctx->d_bias, 1),
                 "conv cublasSaxpy bias");
    conv2d_gpu_zero_gradients(ctx);
}

void conv2d_gpu_reset_adam(Conv2DLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    size_t weight_bytes = (size_t) ctx->weight_elements * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->out_channels * sizeof(nn_float);
    if (ctx->d_m_weights != NULL) {
        check_cuda(cudaMemset(ctx->d_m_weights, 0, weight_bytes),
                   "cudaMemset conv m_weights");
    }
    if (ctx->d_v_weights != NULL) {
        check_cuda(cudaMemset(ctx->d_v_weights, 0, weight_bytes),
                   "cudaMemset conv v_weights");
    }
    if (ctx->d_m_bias != NULL) {
        check_cuda(cudaMemset(ctx->d_m_bias, 0, bias_bytes), "cudaMemset conv m_bias");
    }
    if (ctx->d_v_bias != NULL) {
        check_cuda(cudaMemset(ctx->d_v_bias, 0, bias_bytes), "cudaMemset conv v_bias");
    }
    ctx->adam_initialized = 0;
}

void dense_gpu_apply_gradients(DenseLayerGpuContext *ctx, nn_float learning_rate,
                              int optimizer_kind, nn_float beta1, nn_float beta2,
                              nn_float epsilon, nn_float weight_decay,
                              nn_float inv_bias_correction1, nn_float inv_bias_correction2,
                              nn_float grad_scale)
{
    if (ctx == NULL) {
        return;
    }
    if ((OptimizerKind) optimizer_kind == OPTIMIZER_ADAMW) {
        dense_gpu_ensure_adam_state(ctx);
        nn_float one_minus_beta1 = 1.0f - beta1;
        nn_float one_minus_beta2 = 1.0f - beta2;
        int weight_elements = ctx->output_size * ctx->input_size;
        int bias_elements = ctx->output_size;
        int block = 256;
        int grid_weights = (weight_elements + block - 1) / block;
        adamw_update_kernel<<<grid_weights, block>>>(ctx->d_weights, ctx->d_grad_weights,
                                                     ctx->d_m_weights, ctx->d_v_weights,
                                                     weight_elements, beta1, beta2,
                                                     one_minus_beta1, one_minus_beta2, epsilon,
                                                     learning_rate, weight_decay,
                                                     inv_bias_correction1, inv_bias_correction2,
                                                     grad_scale);
        check_cuda(cudaGetLastError(), "adamw_update_kernel weights");
        int grid_bias = (bias_elements + block - 1) / block;
        adamw_update_kernel<<<grid_bias, block>>>(ctx->d_bias, ctx->d_grad_bias,
                                                  ctx->d_m_bias, ctx->d_v_bias, bias_elements,
                                                  beta1, beta2, one_minus_beta1, one_minus_beta2,
                                                  epsilon, learning_rate, 0.0f,
                                                  inv_bias_correction1, inv_bias_correction2,
                                                  grad_scale);
        check_cuda(cudaGetLastError(), "adamw_update_kernel bias");
        dense_gpu_zero_gradients(ctx);
        return;
    }

    cublasHandle_t handle = global_handle();
    if (weight_decay != 0.0f) {
        check_cublas(cublasSaxpy(handle, ctx->output_size * ctx->input_size, &weight_decay,
                                  ctx->d_weights, 1, ctx->d_grad_weights, 1),
                     "cublasSaxpy weight decay");
    }
    nn_float alpha = -learning_rate;
    check_cublas(cublasSaxpy(handle, ctx->output_size * ctx->input_size, &alpha,
                              ctx->d_grad_weights, 1, ctx->d_weights, 1),
                 "cublasSaxpy weights");
    check_cublas(cublasSaxpy(handle, ctx->output_size, &alpha, ctx->d_grad_bias, 1, ctx->d_bias, 1),
                 "cublasSaxpy bias");
    dense_gpu_zero_gradients(ctx);
}

static __device__ nn_float activation_apply_forward(ActivationKind kind, nn_float x)
{
    switch (kind) {
    case ACT_SIGMOID:
        return 1.0f / (1.0f + expf(-x));
    case ACT_RELU:
        return x > 0.0f ? x : 0.0f;
    case ACT_TANH:
        return tanhf(x);
    case ACT_SOFTMAX:
    default:
        return x;
    }
}

static __device__ nn_float activation_apply_backward(ActivationKind kind, nn_float activated,
                                                      nn_float pre)
{
    (void) pre;
    switch (kind) {
    case ACT_SIGMOID:
        return activated * (1.0f - activated);
    case ACT_RELU:
        return pre > 0.0f ? 1.0f : 0.0f;
    case ACT_TANH:
        return 1.0f - activated * activated;
    case ACT_SOFTMAX:
    default:
        return 1.0f;
    }
}

static __global__ void activation_forward_kernel(nn_float *data, int total, ActivationKind kind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    data[idx] = activation_apply_forward(kind, data[idx]);
}

static __global__ void activation_backward_kernel(const nn_float *activated, nn_float *grad,
                                                   int total, ActivationKind kind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    nn_float derivative = activation_apply_backward(kind, activated[idx], activated[idx]);
    grad[idx] *= derivative;
}

void activation_gpu_forward(int kind_value, nn_float *d_data, int rows, int cols)
{
    int total = rows * cols;
    if (total <= 0) {
        return;
    }
    int block = 256;
    int grid = (total + block - 1) / block;
    activation_forward_kernel<<<grid, block>>>(d_data, total, (ActivationKind) kind_value);
    check_cuda(cudaGetLastError(), "activation_forward_kernel");
}

void activation_gpu_backward(int kind_value, const nn_float *d_output, nn_float *d_grad,
                             int rows, int cols)
{
    int total = rows * cols;
    if (total <= 0) {
        return;
    }
    int block = 256;
    int grid = (total + block - 1) / block;
    activation_backward_kernel<<<grid, block>>>(d_output, d_grad, total, (ActivationKind) kind_value);
    check_cuda(cudaGetLastError(), "activation_backward_kernel");
}

static __device__ float dropout_rand(unsigned long long seed, int idx)
{
    unsigned long long state = seed + (unsigned long long) idx * 0x9E3779B185EBCA87ULL;
    state = (state ^ (state >> 12)) * 0x4BEF9B3F65A6E65ULL;
    state = (state ^ (state >> 25)) * 0x27D4EB2D165667C5ULL;
    state ^= state >> 33;
    unsigned int value = (unsigned int) (state & 0x00FFFFFFu);
    return (float) value / 16777216.0f;
}

static __global__ void dropout_forward_kernel(nn_float *data, nn_float *mask, int elements,
                                              nn_float rate, nn_float keep, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    float r = dropout_rand(seed, idx);
    nn_float retain = (r >= rate) ? 1.0f : 0.0f;
    nn_float scale = (keep > 0.0f) ? retain / keep : 0.0f;
    mask[idx] = scale;
    data[idx] *= scale;
}

static __global__ void dropout_backward_kernel(nn_float *grad, const nn_float *mask, int elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    grad[idx] *= mask[idx];
}

void dropout_gpu_forward(nn_float *d_data, nn_float *d_mask, int elements, nn_float rate,
                         unsigned long long seed)
{
    if (elements <= 0) {
        return;
    }
    nn_float keep = 1.0f - rate;
    int block = 256;
    int grid = (elements + block - 1) / block;
    dropout_forward_kernel<<<grid, block>>>(d_data, d_mask, elements, rate, keep, seed);
    check_cuda(cudaGetLastError(), "dropout_forward_kernel");
}

void dropout_gpu_backward(nn_float *d_grad, const nn_float *d_mask, int elements)
{
    if (elements <= 0) {
        return;
    }
    int block = 256;
    int grid = (elements + block - 1) / block;
    dropout_backward_kernel<<<grid, block>>>(d_grad, d_mask, elements);
    check_cuda(cudaGetLastError(), "dropout_backward_kernel");
}

static __global__ void softmax_forward_kernel(nn_float *data, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    int base = col * rows;
    nn_float max_val = data[base];
    for (int r = 1; r < rows; r++) {
        nn_float v = data[base + r];
        if (v > max_val) {
            max_val = v;
        }
    }
    nn_float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
        nn_float v = expf(data[base + r] - max_val);
        data[base + r] = v;
        sum += v;
    }
    nn_float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int r = 0; r < rows; r++) {
        data[base + r] *= inv;
    }
}

static __global__ void softmax_backward_kernel(const nn_float *output, const nn_float *grad_output,
                                               nn_float *grad_input, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    int base = col * rows;
    for (int i = 0; i < rows; i++) {
        nn_float yi = output[base + i];
        nn_float sum = 0.0f;
        for (int j = 0; j < rows; j++) {
            nn_float yj = output[base + j];
            nn_float derivative = (i == j) ? yi * (1.0f - yi) : -yi * yj;
            sum += derivative * grad_output[base + j];
        }
        grad_input[base + i] = sum;
    }
}

void softmax_gpu_forward(nn_float *d_data, int rows, int cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    int block = 128;
    int grid = (cols + block - 1) / block;
    softmax_forward_kernel<<<grid, block>>>(d_data, rows, cols);
    check_cuda(cudaGetLastError(), "softmax_forward_kernel");
}

void softmax_gpu_backward(const nn_float *d_output, const nn_float *d_grad_output,
                          nn_float *d_grad_input, int rows, int cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    int block = 128;
    int grid = (cols + block - 1) / block;
    softmax_backward_kernel<<<grid, block>>>(d_output, d_grad_output, d_grad_input, rows, cols);
    check_cuda(cudaGetLastError(), "softmax_backward_kernel");
}

static __global__ void batchnorm_compute_stats_kernel(const nn_float *input,
                                                    int channels, int spatial, int batch,
                                                    nn_float *mean, nn_float *var)
{
    extern __shared__ nn_float shared[];
    nn_float *sum = shared;
    nn_float *sumsq = shared + blockDim.x;
    int c = blockIdx.x;
    if (c >= channels) {
        return;
    }
    nn_float local_sum = 0.0f;
    nn_float local_sumsq = 0.0f;
    int rows = channels * spatial;
    int elements = spatial * batch;
    for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
        int col = idx / spatial;
        int offset = idx % spatial;
        int row = c * spatial + offset;
        nn_float v = input[col * rows + row];
        local_sum += v;
        local_sumsq += v * v;
    }
    sum[threadIdx.x] = local_sum;
    sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
            sumsq[threadIdx.x] += sumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        nn_float mean_val = sum[0] / (nn_float) elements;
        nn_float var_val = sumsq[0] / (nn_float) elements - mean_val * mean_val;
        if (var_val < 0.0f) {
            var_val = 0.0f;
        }
        mean[c] = mean_val;
        var[c] = var_val;
    }
}

static __global__ void batchnorm_update_running_kernel(nn_float *running_mean,
                                                       nn_float *running_var,
                                                       const nn_float *batch_mean,
                                                       const nn_float *batch_var,
                                                       int channels, nn_float momentum)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    nn_float mean = batch_mean[c];
    nn_float var = batch_var[c];
    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
    running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
}

static __global__ void batchnorm_forward_kernel(const nn_float *input, nn_float *output,
                                                 const nn_float *gamma, const nn_float *beta,
                                                 const nn_float *mean, const nn_float *var,
                                                 nn_float *normalized, int channels,
                                                 int spatial, int batch, nn_float epsilon)
{
    int rows = channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * batch;
    if (idx >= total) {
        return;
    }
    int col = idx / rows;
    int row = idx % rows;
    int channel = row / spatial;
    nn_float inv_std = rsqrtf(var[channel] + epsilon);
    nn_float value = input[col * rows + row];
    nn_float norm = (value - mean[channel]) * inv_std;
    if (normalized != NULL) {
        normalized[col * rows + row] = norm;
    }
    output[col * rows + row] = gamma[channel] * norm + beta[channel];
}

static __global__ void batchnorm_backward_reduce_kernel(const nn_float *grad_out,
                                                        const nn_float *normalized,
                                                        nn_float *grad_gamma,
                                                        nn_float *grad_beta,
                                                        int channels, int spatial,
                                                        int batch)
{
    extern __shared__ nn_float shared[];
    nn_float *sum_dy = shared;
    nn_float *sum_dy_norm = shared + blockDim.x;
    int c = blockIdx.x;
    if (c >= channels) {
        return;
    }
    int rows = channels * spatial;
    int elements = spatial * batch;
    nn_float local_sum_dy = 0.0f;
    nn_float local_sum_dy_norm = 0.0f;
    for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
        int col = idx / spatial;
        int offset = idx % spatial;
        int row = c * spatial + offset;
        nn_float dy = grad_out[col * rows + row];
        nn_float norm = normalized[col * rows + row];
        local_sum_dy += dy;
        local_sum_dy_norm += dy * norm;
    }
    sum_dy[threadIdx.x] = local_sum_dy;
    sum_dy_norm[threadIdx.x] = local_sum_dy_norm;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_dy[threadIdx.x] += sum_dy[threadIdx.x + stride];
            sum_dy_norm[threadIdx.x] += sum_dy_norm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grad_beta[c] = sum_dy[0];
        grad_gamma[c] = sum_dy_norm[0];
    }
}

static __global__ void batchnorm_backward_input_kernel(const nn_float *grad_out,
                                                        const nn_float *gamma,
                                                        const nn_float *saved_mean,
                                                        const nn_float *saved_var,
                                                        const nn_float *normalized,
                                                        const nn_float *grad_gamma,
                                                        const nn_float *grad_beta,
                                                        nn_float *grad_in,
                                                        int channels, int spatial,
                                                        int batch, nn_float epsilon)
{
    int rows = channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * batch;
    if (idx >= total) {
        return;
    }
    int col = idx / rows;
    int row = idx % rows;
    int channel = row / spatial;
    int elements = spatial * batch;
    nn_float inv_std = rsqrtf(saved_var[channel] + epsilon);
    nn_float gamma_val = gamma[channel];
    nn_float sum_dy = grad_beta[channel];
    nn_float sum_dy_norm = grad_gamma[channel];
    nn_float dy = grad_out[col * rows + row];
    nn_float norm = normalized[col * rows + row];
    nn_float term = (nn_float) elements * dy - sum_dy - norm * sum_dy_norm;
    grad_in[col * rows + row] = gamma_val * inv_std * term / (nn_float) elements;
}

void batchnorm_gpu_forward(nn_float *d_input, nn_float *d_output, const nn_float *d_gamma,
                           const nn_float *d_beta, nn_float *d_running_mean,
                           nn_float *d_running_var, nn_float *d_saved_mean,
                           nn_float *d_saved_var, nn_float *d_normalized, int channels,
                           int spatial, int batch, int training, nn_float momentum,
                           nn_float epsilon)
{
    int rows = channels * spatial;
    int block = 256;
    size_t shared = (size_t) block * 2 * sizeof(nn_float);
    if (training) {
        batchnorm_compute_stats_kernel<<<channels, block, shared>>>(d_input, channels,
                                                                    spatial, batch,
                                                                    d_saved_mean,
                                                                    d_saved_var);
        check_cuda(cudaGetLastError(), "batchnorm_compute_stats_kernel");
        int grid = (channels + 255) / 256;
        if (grid == 0) {
            grid = 1;
        }
        batchnorm_update_running_kernel<<<grid, 256>>>(d_running_mean, d_running_var,
                                                       d_saved_mean, d_saved_var,
                                                       channels, momentum);
        check_cuda(cudaGetLastError(), "batchnorm_update_running_kernel");
    }

    const nn_float *mean = training ? d_saved_mean : d_running_mean;
    const nn_float *var = training ? d_saved_var : d_running_var;
    int total = rows * batch;
    int grid = (total + block - 1) / block;
    batchnorm_forward_kernel<<<grid, block>>>(d_input, d_output, d_gamma, d_beta, mean, var,
                                              training ? d_normalized : NULL, channels,
                                              spatial, batch, epsilon);
    check_cuda(cudaGetLastError(), "batchnorm_forward_kernel");
}

void batchnorm_gpu_backward(const nn_float *d_grad_out, nn_float *d_grad_in,
                            const nn_float *d_gamma, const nn_float *d_saved_mean,
                            const nn_float *d_saved_var, const nn_float *d_normalized,
                            nn_float *d_grad_gamma, nn_float *d_grad_beta, int channels,
                            int spatial, int batch, nn_float epsilon)
{
    size_t bytes = (size_t) channels * sizeof(nn_float);
    check_cuda(cudaMemset(d_grad_gamma, 0, bytes), "cudaMemset bn grad_gamma");
    check_cuda(cudaMemset(d_grad_beta, 0, bytes), "cudaMemset bn grad_beta");

    int block = 256;
    size_t shared = (size_t) block * 2 * sizeof(nn_float);
    batchnorm_backward_reduce_kernel<<<channels, block, shared>>>(
        d_grad_out, d_normalized, d_grad_gamma, d_grad_beta, channels, spatial, batch);
    check_cuda(cudaGetLastError(), "batchnorm_backward_reduce_kernel");

    int rows = channels * spatial;
    int total = rows * batch;
    int grid = (total + block - 1) / block;
    batchnorm_backward_input_kernel<<<grid, block>>>(d_grad_out, d_gamma, d_saved_mean,
                                                     d_saved_var, d_normalized,
                                                     d_grad_gamma, d_grad_beta, d_grad_in,
                                                     channels, spatial, batch, epsilon);
    check_cuda(cudaGetLastError(), "batchnorm_backward_input_kernel");
}

void batchnorm_gpu_zero_gradients(nn_float *d_grad_gamma, nn_float *d_grad_beta,
                                  int channels)
{
    size_t bytes = (size_t) channels * sizeof(nn_float);
    if (d_grad_gamma != NULL) {
        check_cuda(cudaMemset(d_grad_gamma, 0, bytes), "cudaMemset bn grad_gamma");
    }
    if (d_grad_beta != NULL) {
        check_cuda(cudaMemset(d_grad_beta, 0, bytes), "cudaMemset bn grad_beta");
    }
}

static __global__ void vector_add_inplace_kernel(nn_float *output, const nn_float *other, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] += other[idx];
}

void vector_add_inplace(nn_float *d_output, const nn_float *d_other, int elements)
{
    if (elements <= 0) {
        return;
    }
    int block = 256;
    int grid = (elements + block - 1) / block;
    vector_add_inplace_kernel<<<grid, block>>>(d_output, d_other, elements);
    check_cuda(cudaGetLastError(), "vector_add_inplace_kernel");
}

static __global__ void vector_sub_inplace_kernel(nn_float *output, const nn_float *target, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] -= target[idx];
}

void vector_subtract_inplace(nn_float *d_output, const nn_float *d_target, int elements)
{
    if (elements <= 0) {
        return;
    }
    int block = 256;
    int grid = (elements + block - 1) / block;
    vector_sub_inplace_kernel<<<grid, block>>>(d_output, d_target, elements);
    check_cuda(cudaGetLastError(), "vector_sub_inplace_kernel");
}

static __global__ void l2_norm_kernel(const nn_float *data, int n, nn_float *result)
{
    __shared__ nn_float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    nn_float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        nn_float v = data[i];
        sum += v * v;
    }
    cache[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cache[threadIdx.x] += cache[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, cache[0]);
    }
}

nn_float gpu_vector_l2_norm(const nn_float *d_vector, int elements)
{
    if (elements <= 0) {
        return 0.0f;
    }
    nn_float *d_sum = NULL;
    check_cuda(cudaMalloc((void **) &d_sum, sizeof(nn_float)), "cudaMalloc l2 sum");
    check_cuda(cudaMemset(d_sum, 0, sizeof(nn_float)), "cudaMemset l2 sum");

    int block = 256;
    int grid = (elements + block - 1) / block;
    if (grid > 1024) {
        grid = 1024;
    }
    l2_norm_kernel<<<grid, block>>>(d_vector, elements, d_sum);
    check_cuda(cudaGetLastError(), "l2_norm_kernel");

    nn_float host_sum = 0.0f;
    check_cuda(cudaMemcpy(&host_sum, d_sum, sizeof(nn_float), cudaMemcpyDeviceToHost),
               "cudaMemcpy l2 sum");
    check_cuda(cudaFree(d_sum), "cudaFree l2 sum");

    return sqrtf(host_sum);
}

#ifdef __cplusplus
}
#endif

#endif
