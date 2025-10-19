#ifndef AUGMENT_H
#define AUGMENT_H

#include "nn_types.h"
#include "matrix.h"

typedef enum {
    AUGMENT_LAYOUT_NHWC = 0,
    AUGMENT_LAYOUT_NCHW = 1
} AugmentInputLayout;

typedef struct {
    int output_width;
    int output_height;
    int channels;
    int enable_random_resized_crop;
    float scale_min;
    float scale_max;
    float ratio_min;
    float ratio_max;
    int enable_center_crop;
    float center_crop_fraction;
    int enable_horizontal_flip;
    float horizontal_flip_prob;
    int enable_color_jitter;
    float jitter_brightness;
    float jitter_contrast;
    float jitter_saturation;
    int enable_normalize;
    nn_float mean[4];
    nn_float std[4];
} AugmentationConfig;

void augmentation_config_imagenet_train(AugmentationConfig *cfg, int width, int height,
                                        int channels);
void augmentation_config_imagenet_eval(AugmentationConfig *cfg, int width, int height,
                                       int channels);
void augmentation_config_identity(AugmentationConfig *cfg, int width, int height, int channels,
                                  int normalize);

int augmentation_apply_uint8_to_matrix(const unsigned char *src, AugmentInputLayout layout,
                                       int width, int height, int channels,
                                       const AugmentationConfig *cfg, Matrix *dest,
                                       int column);

#endif /* AUGMENT_H */
