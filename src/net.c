#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "internals.h"
#include "matrix.h"
#include "net_internal.h"

#ifdef USE_CUDA
#include "gpu_backend.h"
#endif

void nn_set_training(Network *nn, int training)
{
    if (nn == NULL) {
        return;
    }
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        if (layer->type == LAYER_DROPOUT) {
            layer_dropout_set_training(layer, training);
        } else if (layer->type == LAYER_BATCHNORM) {
            layer_batchnorm_set_training(layer, training);
        }
    }
}

void nn_set_seed(Network *nn, unsigned long long seed)
{
#ifdef USE_CUDA
    if (nn != NULL && nn->backend == BACKEND_GPU && nn->gpu_workspace != NULL) {
        gpu_workspace_set_seed(nn->gpu_workspace, seed);
        return;
    }
#endif
    (void) nn;
    (void) seed;
}

void nn_set_log_memory(Network *nn, int log_memory)
{
    if (nn == NULL) {
        return;
    }
    nn->log_memory = log_memory ? 1 : 0;
}

void nn_set_gpu_theoretical_tflops(Network *nn, double tflops)
{
#ifdef USE_CUDA
    if (nn == NULL || nn->backend != BACKEND_GPU || nn->gpu_workspace == NULL) {
        return;
    }
    gpu_workspace_set_theoretical_tflops(nn->gpu_workspace, tflops);
#else
    (void) nn;
    (void) tflops;
#endif
}

Network *nn_create_empty(int input_size, int output_size, BackendKind backend)
{
    if (input_size <= 0 || output_size <= 0) {
        return NULL;
    }

    Network *nn = (Network *) calloc(1, sizeof(Network));
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
    nn->optimizer.kind = OPTIMIZER_SGD;
    nn->optimizer.beta1 = 0.9f;
    nn->optimizer.beta2 = 0.999f;
    nn->optimizer.epsilon = 1e-8f;
    nn->optimizer.weight_decay = 0.0f;
    nn->optimizer.step = 0;
    nn->optimizer.beta1_power = 1.0f;
    nn->optimizer.beta2_power = 1.0f;
#ifdef USE_CUDA
    nn->gpu_workspace = NULL;
    if (backend == BACKEND_GPU) {
        nn->gpu_workspace = gpu_workspace_create();
        if (nn->gpu_workspace == NULL) {
            free(nn->layers);
            free(nn);
            return NULL;
        }
    }
#endif
    nn->flops_per_sample = 0.0;
    nn->flops_dirty = 1;
    nn->log_memory = 0;

    return nn;
}

Network *nn_create_mlp(int input_size, int hidden_size, int output_size,
                       ActivationKind hidden_activation,
                       ActivationKind output_activation, nn_float dropout_rate,
                       BackendKind backend)
{
    Network *nn = nn_create_empty(input_size, output_size, backend);
    if (nn == NULL) {
        return NULL;
    }

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
        if (dense_ih != NULL) {
            dense_ih->destroy(dense_ih);
            free(dense_ih);
        }
        if (act_hidden != NULL) {
            act_hidden->destroy(act_hidden);
            free(act_hidden);
        }
        if (dropout != NULL) {
            dropout->destroy(dropout);
            free(dropout);
        }
        if (dense_ho != NULL) {
            dense_ho->destroy(dense_ho);
            free(dense_ho);
        }
        if (act_output != NULL) {
            act_output->destroy(act_output);
            free(act_output);
        }
#ifdef USE_CUDA
        if (nn->gpu_workspace != NULL) {
            gpu_workspace_destroy(nn->gpu_workspace);
        }
#endif
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
                         0.0f, BACKEND_CPU);
}

Network *nn_create_with_backend(int input_size, int hidden_size, int output_size,
                                BackendKind backend)
{
    return nn_create_mlp(input_size, hidden_size, output_size, ACT_SIGMOID, ACT_SIGMOID,
                         0.0f, backend);
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
    nn->flops_per_sample = compute_network_flops_per_sample(nn);
    nn->flops_dirty = 0;
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
#ifdef USE_CUDA
    if (nn->gpu_workspace != NULL) {
        gpu_workspace_destroy(nn->gpu_workspace);
    }
#endif
    free(nn);
}

void nn_print(Network *nn)
{
    if (nn == NULL) {
        return;
    }
    printf("Neural Network (%s backend)\n", backend_kind_name(nn->backend));
    printf("Input size: %d\n", nn->input_size);
    printf("Output size: %d\n", nn->output_size);
    printf("Layers (%zu):\n", nn->layer_count);
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            printf("  [%zu] Dense: %d -> %d\n", i, data->weights->cols,
                   data->weights->rows);
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            printf(
                "  [%zu] Conv2D: in_channels=%d, out_channels=%d, input=%dx%d, "
                "kernel=%dx%d, stride=%dx%d, pad=%dx%d\n",
                i, data->in_channels, data->out_channels, data->input_height,
                data->input_width, data->kernel_h, data->kernel_w, data->stride_h,
                data->stride_w, data->pad_h, data->pad_w);
            break;
        }
        case LAYER_ACTIVATION: {
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            const char *name = activation_kind_name(data->kind);
            printf("  [%zu] Activation: %s\n", i, name);
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            printf("  [%zu] Dropout: rate=%.2f training=%s\n", i, data->rate,
                   data->training ? "yes" : "no");
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            printf("  [%zu] BatchNorm: channels=%d spatial=%d training=%s\n", i,
                   data->channels, data->spatial, data->training ? "yes" : "no");
            break;
        }
        case LAYER_SOFTMAX:
            printf("  [%zu] Softmax\n", i);
            break;
        case LAYER_MAXPOOL:
            printf("  [%zu] MaxPool\n", i);
            break;
        case LAYER_GLOBAL_AVG_POOL:
            printf("  [%zu] GlobalAvgPool\n", i);
            break;
        case LAYER_SKIP_SAVE:
            printf("  [%zu] SkipSave\n", i);
            break;
        case LAYER_SKIP_ADD:
            printf("  [%zu] SkipAdd\n", i);
            break;
        }
    }
}

void nn_print_architecture(Network *nn)
{
    if (nn == NULL) {
        return;
    }

    printf("Model:\n");
    printf("  Sequential(\n");
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        printf("    (%zu): ", i);
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            int in_features = data->weights->cols;
            int out_features = data->weights->rows;
            printf("Linear(in_features=%d, out_features=%d, bias=%s)\n", in_features,
                   out_features, data->bias != NULL ? "True" : "False");
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            printf(
                "Conv2d(in_channels=%d, out_channels=%d, kernel_size=(%d, %d), "
                "stride=(%d, %d), padding=(%d, %d))\n",
                data->in_channels, data->out_channels, data->kernel_h, data->kernel_w,
                data->stride_h, data->stride_w, data->pad_h, data->pad_w);
            break;
        }
        case LAYER_ACTIVATION: {
            ActivationLayerData *data = (ActivationLayerData *) layer->data;
            const char *name = activation_kind_name(data->kind);
            printf("%s()\n", name);
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayerData *data = (DropoutLayerData *) layer->data;
            printf("Dropout(p=%.4f, inplace=False)\n", data->rate);
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            printf("BatchNorm(channels=%d, spatial=%d)\n", data->channels, data->spatial);
            break;
        }
        case LAYER_MAXPOOL: {
            MaxPoolLayerData *data = (MaxPoolLayerData *) layer->data;
            printf("MaxPool2d(kernel_size=(%d, %d), stride=(%d, %d), padding=(%d, %d))\n",
                   data->kernel_h, data->kernel_w, data->stride_h, data->stride_w, data->pad_h,
                   data->pad_w);
            break;
        }
        case LAYER_GLOBAL_AVG_POOL:
            printf("AdaptiveAvgPool2d(output_size=(1, 1))\n");
            break;
        case LAYER_SOFTMAX:
            printf("Softmax(dim=1)\n");
            break;
        case LAYER_SKIP_SAVE:
            printf("SkipSave()\n");
            break;
        case LAYER_SKIP_ADD:
            printf("SkipAdd()\n");
            break;
        }
    }
    printf("  )\n");
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
#ifdef USE_CUDA
            if (data->backend == BACKEND_GPU && data->gpu != NULL) {
                dense_gpu_zero_gradients(data->gpu);
            }
#endif
        } else if (layer->type == LAYER_CONV2D) {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            matrix_zero(data->grad_weights);
            matrix_zero(data->grad_bias);
#ifdef USE_CUDA
            if (data->backend == BACKEND_GPU && data->gpu != NULL) {
                conv2d_gpu_zero_gradients(data->gpu);
            }
#endif
        } else if (layer->type == LAYER_BATCHNORM) {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            matrix_zero(data->grad_gamma);
            matrix_zero(data->grad_beta);
#ifdef USE_CUDA
            if (data->backend == BACKEND_GPU) {
                batchnorm_layer_gpu_ensure_capacity(data, 1);
                batchnorm_layer_gpu_copy_params_to_device(data);
                batchnorm_layer_gpu_copy_running_to_device(data);
                batchnorm_gpu_zero_gradients(data->d_grad_gamma, data->d_grad_beta,
                                             data->channels);
            }
#endif
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
    OptimizerState *opt = &nn->optimizer;

    opt->step += 1;
    opt->beta1_power *= opt->beta1;
    opt->beta2_power *= opt->beta2;

    nn_float inv_bias_correction1 = 1.0f;
    nn_float inv_bias_correction2 = 1.0f;
    if (opt->kind == OPTIMIZER_ADAMW) {
        nn_float denom1 = 1.0f - opt->beta1_power;
        nn_float denom2 = 1.0f - opt->beta2_power;
        if (denom1 > 0.0f) {
            inv_bias_correction1 = 1.0f / denom1;
        }
        if (denom2 > 0.0f) {
            inv_bias_correction2 = 1.0f / denom2;
        }
    }

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            if (opt->kind == OPTIMIZER_ADAMW) {
                dense_layer_apply_adamw(data, opt, scaled_lr, inv_bias_correction1,
                                        inv_bias_correction2, 1.0f / (nn_float) batch_size);
            } else {
                dense_layer_apply_sgd(data, scaled_lr, opt->weight_decay);
            }
#ifdef USE_CUDA
            if (data->backend == BACKEND_GPU && data->gpu != NULL) {
                dense_gpu_apply_gradients(data->gpu, scaled_lr, opt->kind, opt->beta1,
                                          opt->beta2, opt->epsilon, opt->weight_decay,
                                          inv_bias_correction1, inv_bias_correction2,
                                          1.0f / (nn_float) batch_size);
            }
#endif
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            if (opt->kind == OPTIMIZER_ADAMW) {
                conv2d_layer_apply_adamw(data, opt, scaled_lr, inv_bias_correction1,
                                          inv_bias_correction2, 1.0f / (nn_float) batch_size);
            } else {
                conv2d_layer_apply_sgd(data, scaled_lr, opt->weight_decay);
            }
#ifdef USE_CUDA
            if (data->backend == BACKEND_GPU && data->gpu != NULL) {
                conv2d_gpu_apply_gradients(data->gpu, scaled_lr, opt->kind, opt->beta1,
                                           opt->beta2, opt->epsilon, opt->weight_decay,
                                           inv_bias_correction1, inv_bias_correction2,
                                           1.0f / (nn_float) batch_size);
            }
#endif
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            if (opt->kind == OPTIMIZER_ADAMW) {
                batchnorm_layer_apply_adamw(data, opt, scaled_lr, inv_bias_correction1,
                                            inv_bias_correction2, 1.0f / (nn_float) batch_size);
            } else {
                batchnorm_layer_apply_sgd(data, scaled_lr, opt->weight_decay,
                                           1.0f / (nn_float) batch_size);
            }
            break;
        }
        default:
            break;
        }
    }
}

void nn_set_optimizer(Network *nn, OptimizerKind kind, nn_float beta1, nn_float beta2,
                      nn_float epsilon, nn_float weight_decay)
{
    if (nn == NULL) {
        return;
    }

    nn->optimizer.kind = kind;
    nn->optimizer.beta1 = beta1;
    nn->optimizer.beta2 = beta2;
    nn->optimizer.epsilon = epsilon;
    nn->optimizer.weight_decay = weight_decay;
    nn->optimizer.step = 0;
    nn->optimizer.beta1_power = 1.0f;
    nn->optimizer.beta2_power = 1.0f;

    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        if (layer->type == LAYER_DENSE) {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            data->adam_initialized = 0;
#ifdef USE_CUDA
            data->adam_device_initialized = 0;
            if (data->gpu != NULL) {
                dense_gpu_reset_adam(data->gpu);
            }
#endif
        } else if (layer->type == LAYER_CONV2D) {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            data->adam_initialized = 0;
#ifdef USE_CUDA
            data->adam_device_initialized = 0;
            if (data->gpu != NULL) {
                conv2d_gpu_reset_adam(data->gpu);
            }
#endif
        } else if (layer->type == LAYER_BATCHNORM) {
            BatchNormLayerData *data = (BatchNormLayerData *) layer->data;
            data->adam_initialized = 0;
        }
    }
}

TrainStepStats nn_train_batch(Network *nn, Matrix *input, Matrix *target,
                              int batch_size, Matrix *output_buffer)
{
#ifdef USE_CUDA
    if (nn->backend == BACKEND_GPU) {
        return nn_train_batch_gpu(nn, input, target, batch_size, output_buffer);
    }
#endif
    return nn_train_batch_cpu(nn, input, target, batch_size, output_buffer);
}
