#include "dataset.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

typedef struct {
    nn_float inputs[4][2];
    nn_float outputs[4];
} XorData;

typedef struct {
    int rows;
    int cols;
    int sample_count;
    unsigned char *images;
    unsigned char *labels;
} MnistData;

static Dataset *dataset_create(void)
{
    Dataset *dataset = (Dataset *) malloc(sizeof(Dataset));
    if (dataset == NULL) {
        return NULL;
    }
    dataset->data = NULL;
    dataset->input_size = 0;
    dataset->output_size = 0;
    dataset->sample_count = 0;
    dataset->kind = DATASET_XOR;
    return dataset;
}

Dataset *dataset_create_xor(void)
{
    Dataset *dataset = dataset_create();
    if (dataset == NULL) {
        return NULL;
    }

    XorData *data = (XorData *) malloc(sizeof(XorData));
    if (data == NULL) {
        free(dataset);
        return NULL;
    }

    nn_float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    nn_float outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    memcpy(data->inputs, inputs, sizeof(inputs));
    memcpy(data->outputs, outputs, sizeof(outputs));

    dataset->kind = DATASET_XOR;
    dataset->input_size = 2;
    dataset->output_size = 1;
    dataset->sample_count = 4;
    dataset->data = data;

    return dataset;
}

static uint32_t read_be_u32(FILE *file, int *ok)
{
    unsigned char buffer[4];
    if (fread(buffer, 1, 4, file) != 4) {
        *ok = 0;
        return 0;
    }
    uint32_t value = ((uint32_t) buffer[0] << 24) | ((uint32_t) buffer[1] << 16) |
                     ((uint32_t) buffer[2] << 8) | (uint32_t) buffer[3];
    return value;
}

Dataset *dataset_load_mnist(const char *images_path, const char *labels_path, int limit)
{
    if (images_path == NULL || labels_path == NULL) {
        return NULL;
    }

    FILE *images = fopen(images_path, "rb");
    if (images == NULL) {
        return NULL;
    }
    FILE *labels = fopen(labels_path, "rb");
    if (labels == NULL) {
        fclose(images);
        return NULL;
    }

    int ok = 1;
    uint32_t magic_images = read_be_u32(images, &ok);
    uint32_t image_count = read_be_u32(images, &ok);
    uint32_t rows = read_be_u32(images, &ok);
    uint32_t cols = read_be_u32(images, &ok);

    uint32_t magic_labels = read_be_u32(labels, &ok);
    uint32_t label_count = read_be_u32(labels, &ok);

    if (!ok || magic_images != 2051 || magic_labels != 2049 || image_count != label_count ||
        rows == 0 || cols == 0) {
        fclose(images);
        fclose(labels);
        return NULL;
    }

    if (limit > 0 && (uint32_t) limit < image_count) {
        image_count = (uint32_t) limit;
    }

    size_t image_size = (size_t) rows * (size_t) cols;
    size_t total_image_bytes = image_size * image_count;

    MnistData *data = (MnistData *) malloc(sizeof(MnistData));
    if (data == NULL) {
        fclose(images);
        fclose(labels);
        return NULL;
    }

    data->rows = (int) rows;
    data->cols = (int) cols;
    data->sample_count = (int) image_count;
    data->images = (unsigned char *) malloc(total_image_bytes);
    data->labels = (unsigned char *) malloc(image_count);

    if (data->images == NULL || data->labels == NULL) {
        free(data->images);
        free(data->labels);
        free(data);
        fclose(images);
        fclose(labels);
        return NULL;
    }

    if (fread(data->images, 1, total_image_bytes, images) != total_image_bytes ||
        fread(data->labels, 1, image_count, labels) != image_count) {
        free(data->images);
        free(data->labels);
        free(data);
        fclose(images);
        fclose(labels);
        return NULL;
    }

    fclose(images);
    fclose(labels);

    Dataset *dataset = dataset_create();
    if (dataset == NULL) {
        free(data->images);
        free(data->labels);
        free(data);
        return NULL;
    }

    dataset->kind = DATASET_MNIST;
    dataset->input_size = data->rows * data->cols;
    dataset->output_size = 10;
    dataset->sample_count = data->sample_count;
    dataset->data = data;

    return dataset;
}

void dataset_free(Dataset *dataset)
{
    if (dataset == NULL) {
        return;
    }

    if (dataset->data != NULL) {
        if (dataset->kind == DATASET_XOR) {
            free(dataset->data);
        } else if (dataset->kind == DATASET_MNIST) {
            MnistData *mnist = (MnistData *) dataset->data;
            free(mnist->images);
            free(mnist->labels);
            free(mnist);
        }
    }

    free(dataset);
}

void dataset_fill_sample_column(const Dataset *dataset, int index, Matrix *input,
                                Matrix *target, int column)
{
    if (dataset == NULL || input == NULL || target == NULL) {
        return;
    }
    if (index < 0 || index >= dataset->sample_count) {
        return;
    }
    if (column < 0 || column >= input->cols || column >= target->cols) {
        return;
    }
    if (input->rows != dataset->input_size || target->rows != dataset->output_size) {
        return;
    }

    if (dataset->kind == DATASET_XOR) {
        XorData *xor_data = (XorData *) dataset->data;
        for (int r = 0; r < dataset->input_size; r++) {
            input->data[r * input->cols + column] = xor_data->inputs[index][r];
        }
        for (int r = 0; r < dataset->output_size; r++) {
            target->data[r * target->cols + column] = xor_data->outputs[index];
        }
        return;
    } else if (dataset->kind == DATASET_MNIST) {
        MnistData *mnist = (MnistData *) dataset->data;
        size_t offset = (size_t) index * (size_t) dataset->input_size;
        for (int i = 0; i < dataset->input_size; i++) {
            input->data[i * input->cols + column] =
                (nn_float) mnist->images[offset + (size_t) i] / 255.0f;
        }
        for (int i = 0; i < dataset->output_size; i++) {
            target->data[i * target->cols + column] = 0.0f;
        }
        int label = mnist->labels[index];
        if (label >= 0 && label < dataset->output_size) {
            target->data[label * target->cols + column] = 1.0f;
        }
        return;
    }

    return;
}

int dataset_fill_sample(const Dataset *dataset, int index, Matrix *input, Matrix *target)
{
    if (input == NULL || target == NULL || input->cols != 1 || target->cols != 1) {
        return 0;
    }
    dataset_fill_sample_column(dataset, index, input, target, 0);
    return 1;
}

int dataset_get_label(const Dataset *dataset, int index)
{
    if (dataset == NULL || index < 0 || index >= dataset->sample_count) {
        return -1;
    }

    if (dataset->kind == DATASET_XOR) {
        XorData *xor_data = (XorData *) dataset->data;
        return xor_data->outputs[index] > 0.5f ? 1 : 0;
    }

    MnistData *mnist = (MnistData *) dataset->data;
    return mnist->labels[index];
}

const char *dataset_kind_name(DatasetKind kind)
{
    switch (kind) {
    case DATASET_XOR:
        return "xor";
    case DATASET_MNIST:
        return "mnist";
    }
    return "unknown";
}
