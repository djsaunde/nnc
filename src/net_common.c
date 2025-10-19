#include <math.h>

#include "net_internal.h"

static void copy_matrix_to_column_major_internal(const Matrix *src, nn_float *dst)
{
    int rows = src->rows;
    int cols = src->cols;
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            dst[row + col * rows] = src->data[row * src->cols + col];
        }
    }
}

static void copy_column_major_to_matrix_internal(Matrix *dst, const nn_float *src)
{
    int rows = dst->rows;
    int cols = dst->cols;
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            dst->data[row * dst->cols + col] = src[row + col * rows];
        }
    }
}

double compute_network_flops_per_sample(Network *nn)
{
    if (nn == NULL) {
        return 0.0;
    }

    double total = 0.0;
    for (size_t i = 0; i < nn->layer_count; i++) {
        Layer *layer = nn->layers[i];
        switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayerData *data = (DenseLayerData *) layer->data;
            if (data != NULL && data->weights != NULL) {
                double m = (double) data->weights->rows;
                double k = (double) data->weights->cols;
                double forward = 2.0 * m * k;
                double backward = 4.0 * m * k;
                double bias = 2.0 * m;
                total += forward + backward + bias;
            }
            break;
        }
        case LAYER_CONV2D: {
            Conv2DLayerData *data = (Conv2DLayerData *) layer->data;
            if (data != NULL) {
                double out_spatial = (double) data->output_height * data->output_width;
                double kernel_area = (double) data->kernel_h * data->kernel_w;
                double oc = (double) data->out_channels;
                double ic = (double) data->in_channels;
                double forward = 2.0 * oc * out_spatial * ic * kernel_area;
                double backward = 4.0 * oc * out_spatial * ic * kernel_area;
                double bias = 2.0 * oc * out_spatial;
                total += forward + backward + bias;
            }
            break;
        }
        default:
            break;
        }
    }

    return total;
}

double compute_loss_from_output(Network *nn, Matrix *output, Matrix *target,
                                int batch_size)
{
    const double epsilon = 1e-7;
    int rows = output->rows;
    int cols = batch_size;
    Layer *output_layer = nn->layer_count > 0 ? nn->layers[nn->layer_count - 1] : NULL;
    int use_softmax = output_layer != NULL && output_layer->type == LAYER_SOFTMAX;

    double loss = 0.0;
    if (use_softmax) {
        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < rows; row++) {
                double y = target->data[row * target->cols + col];
                if (y <= 0.0) {
                    continue;
                }
                double y_hat = output->data[row * output->cols + col];
                y_hat = fmax(epsilon, y_hat);
                loss += -y * log(y_hat);
            }
        }
        loss /= cols;
    } else if (rows == 1) {
        for (int col = 0; col < cols; col++) {
            double y = target->data[col];
            double y_hat = output->data[col];
            y_hat = fmax(epsilon, fmin(1.0 - epsilon, y_hat));
            loss += -(y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat));
        }
        loss /= cols;
    } else {
        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < rows; row++) {
                double y = target->data[row * target->cols + col];
                double y_hat = output->data[row * output->cols + col];
                y_hat = fmax(epsilon, fmin(1.0 - epsilon, y_hat));
                loss += -(y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat));
            }
        }
        loss /= (rows * cols);
    }

    return loss;
}

double nn_estimated_flops_per_sample(Network *nn)
{
    if (nn == NULL) {
        return 0.0;
    }
    if (nn->flops_dirty) {
        nn->flops_per_sample = compute_network_flops_per_sample(nn);
        nn->flops_dirty = 0;
    }
    return nn->flops_per_sample;
}

void copy_matrix_to_column_major(const Matrix *src, nn_float *dst)
{
    copy_matrix_to_column_major_internal(src, dst);
}

void copy_column_major_to_matrix(Matrix *dst, const nn_float *src)
{
    copy_column_major_to_matrix_internal(dst, src);
}
