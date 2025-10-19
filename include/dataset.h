#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"

typedef enum { DATASET_XOR, DATASET_MNIST } DatasetKind;

typedef struct {
    DatasetKind kind;
    int input_size;
    int output_size;
    int sample_count;
    void *data;
} Dataset;

Dataset *dataset_create_xor(void);
Dataset *dataset_load_mnist(const char *images_path, const char *labels_path,
                            int limit);
void dataset_free(Dataset *dataset);
int dataset_fill_sample(const Dataset *dataset, int index, Matrix *input,
                        Matrix *target);
int dataset_get_label(const Dataset *dataset, int index);
const char *dataset_kind_name(DatasetKind kind);
void dataset_fill_sample_column(const Dataset *dataset, int index, Matrix *input,
                                Matrix *target, int column);

#endif
