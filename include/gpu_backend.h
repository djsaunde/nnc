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
void dense_gpu_set_weights(DenseLayerGpuContext *ctx, const nn_float *weights,
                           const nn_float *bias);
void dense_gpu_get_weights(DenseLayerGpuContext *ctx, nn_float *weights,
                           nn_float *bias);
nn_float *dense_gpu_forward_device(DenseLayerGpuContext *ctx, const nn_float *d_input,
                                   int batch);
nn_float *dense_gpu_backward_device(DenseLayerGpuContext *ctx,
                                    const nn_float *d_grad_output, int batch);
void dense_gpu_zero_gradients(DenseLayerGpuContext *ctx);
void dense_gpu_apply_gradients(DenseLayerGpuContext *ctx, nn_float learning_rate,
                               int optimizer_kind, nn_float beta1, nn_float beta2,
                               nn_float epsilon, nn_float weight_decay,
                               nn_float inv_bias_correction1,
                               nn_float inv_bias_correction2, nn_float grad_scale);
void dense_gpu_reset_adam(DenseLayerGpuContext *ctx);

typedef struct Conv2DLayerGpuContext Conv2DLayerGpuContext;
Conv2DLayerGpuContext *conv2d_gpu_create(int in_channels, int out_channels,
                                         int input_height, int input_width,
                                         int kernel_h, int kernel_w, int stride_h,
                                         int stride_w, int pad_h, int pad_w);
void conv2d_gpu_destroy(Conv2DLayerGpuContext *ctx);
void conv2d_gpu_set_weights(Conv2DLayerGpuContext *ctx, const nn_float *weights,
                            const nn_float *bias);
nn_float *conv2d_gpu_forward_device(Conv2DLayerGpuContext *ctx, const nn_float *d_input,
                                    int batch);
nn_float *conv2d_gpu_backward_device(Conv2DLayerGpuContext *ctx,
                                     const nn_float *d_grad_output, int batch);
void conv2d_gpu_zero_gradients(Conv2DLayerGpuContext *ctx);
void conv2d_gpu_apply_gradients(Conv2DLayerGpuContext *ctx, nn_float learning_rate,
                                int optimizer_kind, nn_float beta1, nn_float beta2,
                                nn_float epsilon, nn_float weight_decay,
                                nn_float inv_bias_correction1,
                                nn_float inv_bias_correction2, nn_float grad_scale);
void conv2d_gpu_reset_adam(Conv2DLayerGpuContext *ctx);

void batchnorm_gpu_forward(nn_float *d_input, nn_float *d_output,
                                const nn_float *d_gamma, const nn_float *d_beta,
                                nn_float *d_running_mean, nn_float *d_running_var,
                                nn_float *d_saved_mean, nn_float *d_saved_var,
                                nn_float *d_normalized, int channels, int spatial,
                                int batch, int training, nn_float momentum,
                                nn_float epsilon);
void batchnorm_gpu_backward(const nn_float *d_grad_out, nn_float *d_grad_in,
                            const nn_float *d_gamma, const nn_float *d_saved_mean,
                            const nn_float *d_saved_var, const nn_float *d_normalized,
                            nn_float *d_grad_gamma, nn_float *d_grad_beta,
                            int channels, int spatial, int batch, nn_float epsilon);
void batchnorm_gpu_zero_gradients(nn_float *d_grad_gamma, nn_float *d_grad_beta,
                                  int channels);
void batchnorm_gpu_apply_sgd(nn_float *d_gamma, nn_float *d_beta,
                             nn_float *d_grad_gamma, nn_float *d_grad_beta, int channels,
                             nn_float learning_rate, nn_float weight_decay,
                             nn_float grad_scale);
void batchnorm_gpu_apply_adamw(nn_float *d_gamma, nn_float *d_beta,
                               nn_float *d_grad_gamma, nn_float *d_grad_beta,
                               nn_float *d_m_gamma, nn_float *d_v_gamma,
                               nn_float *d_m_beta, nn_float *d_v_beta, int channels,
                               nn_float learning_rate, nn_float beta1, nn_float beta2,
                               nn_float epsilon, nn_float weight_decay,
                               nn_float inv_bias_correction1,
                               nn_float inv_bias_correction2, nn_float grad_scale);
void activation_gpu_forward(int kind, nn_float *d_data, int rows, int cols);
void activation_gpu_backward(int kind, const nn_float *d_output, nn_float *d_grad,
                             int rows, int cols);

void dropout_gpu_forward(nn_float *d_data, nn_float *d_mask, int elements,
                         nn_float rate, unsigned long long seed);
void dropout_gpu_backward(nn_float *d_grad, const nn_float *d_mask, int elements);

void softmax_gpu_forward(nn_float *d_data, int rows, int cols);
void softmax_gpu_backward(const nn_float *d_output, const nn_float *d_grad_output,
                          nn_float *d_grad_input, int rows, int cols);

void vector_subtract_inplace(nn_float *d_output, const nn_float *d_target,
                             int elements);
void vector_add_inplace(nn_float *d_output, const nn_float *d_other, int elements);
nn_float gpu_vector_l2_norm(const nn_float *d_vector, int elements, nn_float *d_workspace);

#ifdef __cplusplus
}
#endif
#endif

#endif
