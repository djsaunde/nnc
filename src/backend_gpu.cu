#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>

#include "gpu_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DenseLayerGpuContext {
    int input_size;
    int output_size;
    int capacity;
    nn_float *d_weights;
    nn_float *d_bias;
    nn_float *d_input;
    nn_float *d_output;
    nn_float *d_last_input;
    nn_float *d_grad_output;
    nn_float *d_grad_input;
    nn_float *d_grad_weights;
    nn_float *h_input_col;
    nn_float *h_output_col;
    nn_float *h_last_input_col;
    nn_float *h_grad_output_col;
    nn_float *h_grad_input_col;
} DenseLayerGpuContext;

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

static void col_to_row_add(nn_float *dst, const nn_float *src, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[r * cols + c] += src[c * rows + r];
        }
    }
}

static void ensure_capacity(DenseLayerGpuContext *ctx, int batch)
{
    if (batch <= ctx->capacity) {
        return;
    }

    size_t input_bytes = (size_t) ctx->input_size * batch * sizeof(nn_float);
    size_t output_bytes = (size_t) ctx->output_size * batch * sizeof(nn_float);

    if (ctx->d_input != NULL) {
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_output);
        cudaFree(ctx->d_last_input);
        cudaFree(ctx->d_grad_output);
        cudaFree(ctx->d_grad_input);
        free(ctx->h_input_col);
        free(ctx->h_output_col);
        free(ctx->h_last_input_col);
        free(ctx->h_grad_output_col);
        free(ctx->h_grad_input_col);
    }

    check_cuda(cudaMalloc((void **) &ctx->d_input, input_bytes), "cudaMalloc d_input");
    check_cuda(cudaMalloc((void **) &ctx->d_last_input, input_bytes), "cudaMalloc d_last_input");
    check_cuda(cudaMalloc((void **) &ctx->d_output, output_bytes), "cudaMalloc d_output");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_output, output_bytes), "cudaMalloc d_grad_output");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_input, input_bytes), "cudaMalloc d_grad_input");

    ctx->h_input_col = (nn_float *) malloc(input_bytes);
    ctx->h_last_input_col = (nn_float *) malloc(input_bytes);
    ctx->h_output_col = (nn_float *) malloc(output_bytes);
    ctx->h_grad_output_col = (nn_float *) malloc(output_bytes);
    ctx->h_grad_input_col = (nn_float *) malloc(input_bytes);

    ctx->capacity = batch;
}

DenseLayerGpuContext *dense_gpu_create(int input_size, int output_size)
{
    DenseLayerGpuContext *ctx = (DenseLayerGpuContext *) malloc(sizeof(DenseLayerGpuContext));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->input_size = input_size;
    ctx->output_size = output_size;
    ctx->capacity = 0;
    ctx->d_input = NULL;
    ctx->d_output = NULL;
    ctx->d_last_input = NULL;
    ctx->d_grad_output = NULL;
    ctx->d_grad_input = NULL;
    ctx->h_input_col = NULL;
    ctx->h_output_col = NULL;
    ctx->h_last_input_col = NULL;
    ctx->h_grad_output_col = NULL;
    ctx->h_grad_input_col = NULL;

    size_t weight_bytes = (size_t) output_size * input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) output_size * sizeof(nn_float);

    check_cuda(cudaMalloc((void **) &ctx->d_weights, weight_bytes), "cudaMalloc d_weights");
    check_cuda(cudaMalloc((void **) &ctx->d_bias, bias_bytes), "cudaMalloc d_bias");
    check_cuda(cudaMalloc((void **) &ctx->d_grad_weights, weight_bytes), "cudaMalloc d_grad_weights");

    return ctx;
}

void dense_gpu_destroy(DenseLayerGpuContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    cudaFree(ctx->d_weights);
    cudaFree(ctx->d_bias);
    cudaFree(ctx->d_grad_weights);
    if (ctx->d_input != NULL) {
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_output);
        cudaFree(ctx->d_last_input);
        cudaFree(ctx->d_grad_output);
        cudaFree(ctx->d_grad_input);
    }
    free(ctx->h_input_col);
    free(ctx->h_last_input_col);
    free(ctx->h_output_col);
    free(ctx->h_grad_output_col);
    free(ctx->h_grad_input_col);
    free(ctx);
}

void dense_gpu_set_weights(DenseLayerGpuContext *ctx, const nn_float *weights, const nn_float *bias)
{
    size_t weight_bytes = (size_t) ctx->output_size * ctx->input_size * sizeof(nn_float);
    size_t bias_bytes = (size_t) ctx->output_size * sizeof(nn_float);

    nn_float *temp = (nn_float *) malloc(weight_bytes);
    row_to_col(temp, weights, ctx->output_size, ctx->input_size);
    check_cuda(cudaMemcpy(ctx->d_weights, temp, weight_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy d_weights");
    free(temp);

    check_cuda(cudaMemcpy(ctx->d_bias, bias, bias_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy d_bias");
}

void dense_gpu_forward(DenseLayerGpuContext *ctx, const nn_float *input, int batch, nn_float *output,
                       nn_float *last_input)
{
    ensure_capacity(ctx, batch);

    row_to_col(ctx->h_input_col, input, ctx->input_size, batch);
    check_cuda(cudaMemcpy(ctx->d_input, ctx->h_input_col,
                          (size_t) ctx->input_size * batch * sizeof(nn_float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_input");

    row_to_col(ctx->h_last_input_col, last_input, ctx->input_size, batch);
    check_cuda(cudaMemcpy(ctx->d_last_input, ctx->h_last_input_col,
                          (size_t) ctx->input_size * batch * sizeof(nn_float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_last_input");

    const nn_float alpha = 1.0f;
    const nn_float beta = 0.0f;
    cublasHandle_t handle = global_handle();

    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ctx->output_size, batch,
                              ctx->input_size, &alpha, ctx->d_weights, ctx->output_size,
                              ctx->d_input, ctx->input_size, &beta, ctx->d_output,
                              ctx->output_size),
                 "cublasSgemm forward");

    check_cuda(cudaMemcpy(ctx->h_output_col, ctx->d_output,
                          (size_t) ctx->output_size * batch * sizeof(nn_float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    col_to_row(output, ctx->h_output_col, ctx->output_size, batch);
}

void dense_gpu_backward(DenseLayerGpuContext *ctx, const nn_float *grad_output,
                        const nn_float *last_input, int batch, nn_float *grad_weights,
                        nn_float *grad_input)
{
    ensure_capacity(ctx, batch);

    row_to_col(ctx->h_grad_output_col, grad_output, ctx->output_size, batch);
    check_cuda(cudaMemcpy(ctx->d_grad_output, ctx->h_grad_output_col,
                          (size_t) ctx->output_size * batch * sizeof(nn_float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_grad_output");

    row_to_col(ctx->h_last_input_col, last_input, ctx->input_size, batch);
    check_cuda(cudaMemcpy(ctx->d_last_input, ctx->h_last_input_col,
                          (size_t) ctx->input_size * batch * sizeof(nn_float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_last_input");

    const nn_float alpha = 1.0f;
    const nn_float zero = 0.0f;
    cublasHandle_t handle = global_handle();

    check_cuda(cudaMemset(ctx->d_grad_weights, 0,
                          (size_t) ctx->output_size * ctx->input_size * sizeof(nn_float)),
               "cudaMemset grad_weights");

    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ctx->output_size, ctx->input_size,
                              batch, &alpha, ctx->d_grad_output, ctx->output_size,
                              ctx->d_last_input, ctx->input_size, &zero, ctx->d_grad_weights,
                              ctx->output_size),
                 "cublasSgemm grad_weights");

    nn_float *temp =
        (nn_float *) malloc((size_t) ctx->output_size * ctx->input_size * sizeof(nn_float));
    check_cuda(cudaMemcpy(temp, ctx->d_grad_weights,
                          (size_t) ctx->output_size * ctx->input_size * sizeof(nn_float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy grad_weights");
    col_to_row_add(grad_weights, temp, ctx->output_size, ctx->input_size);
    free(temp);

    check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ctx->input_size, batch,
                              ctx->output_size, &alpha, ctx->d_weights, ctx->output_size,
                              ctx->d_grad_output, ctx->output_size, &zero, ctx->d_grad_input,
                              ctx->input_size),
                 "cublasSgemm grad_input");

    check_cuda(cudaMemcpy(ctx->h_grad_input_col, ctx->d_grad_input,
                          (size_t) ctx->input_size * batch * sizeof(nn_float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy grad_input");
    col_to_row(grad_input, ctx->h_grad_input_col, ctx->input_size, batch);
}

void dense_gpu_update(DenseLayerGpuContext *ctx, const nn_float *weights, const nn_float *bias)
{
    dense_gpu_set_weights(ctx, weights, bias);
}

#ifdef __cplusplus
}
#endif

#endif
