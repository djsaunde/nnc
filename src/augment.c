#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "augment.h"

static float random_uniform(float min_value, float max_value)
{
    if (max_value <= min_value) {
        return min_value;
    }
    float t = (float) rand() / (float) RAND_MAX;
    return min_value + (max_value - min_value) * t;
}

static float clamp_float(float value, float min_value, float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static void convert_to_hwc(const unsigned char *src, AugmentInputLayout layout, int width,
                           int height, int channels, float *dst)
{
    if (layout == AUGMENT_LAYOUT_NHWC) {
        int total = width * height * channels;
        for (int i = 0; i < total; i++) {
            dst[i] = (float) src[i] / 255.0f;
        }
        return;
    }

    /* NCHW -> NHWC */
    int plane = width * height;
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int nchw_index = c * plane + y * width + x;
                int hwc_index = (y * width + x) * channels + c;
                dst[hwc_index] = (float) src[nchw_index] / 255.0f;
            }
        }
    }
}

static inline float sample_hwc_pixel(const float *image, int width, int height, int channels,
                                     int x, int y, int c)
{
    if (x < 0) {
        x = 0;
    }
    if (y < 0) {
        y = 0;
    }
    if (x >= width) {
        x = width - 1;
    }
    if (y >= height) {
        y = height - 1;
    }
    int index = (y * width + x) * channels + c;
    return image[index];
}

static void augmentation_select_crop(const AugmentationConfig *cfg, int width, int height,
                                     int *crop_x, int *crop_y, int *crop_w, int *crop_h,
                                     int *flip)
{
    *crop_x = 0;
    *crop_y = 0;
    *crop_w = width;
    *crop_h = height;
    *flip = 0;

    if (cfg == NULL) {
        return;
    }

    if (cfg->enable_random_resized_crop) {
        float area = (float) width * (float) height;
        for (int attempt = 0; attempt < 10; attempt++) {
            float target_area = area * random_uniform(cfg->scale_min, cfg->scale_max);
            float log_min = logf(cfg->ratio_min);
            float log_max = logf(cfg->ratio_max);
            float aspect = expf(random_uniform(log_min, log_max));

            int w = (int) roundf(sqrtf(target_area * aspect));
            int h = (int) roundf(sqrtf(target_area / aspect));
            if (w > 0 && h > 0 && w <= width && h <= height) {
                int x = 0;
                int y = 0;
                if (width > w) {
                    x = rand() % (width - w + 1);
                }
                if (height > h) {
                    y = rand() % (height - h + 1);
                }
                *crop_x = x;
                *crop_y = y;
                *crop_w = w;
                *crop_h = h;
                break;
            }
        }
    } else if (cfg->enable_center_crop) {
        float fraction = clamp_float(cfg->center_crop_fraction, 0.0f, 1.0f);
        if (fraction > 0.0f && fraction < 1.0f) {
            int w = (int) roundf(width * fraction);
            int h = (int) roundf(height * fraction);
            if (w > 0 && h > 0) {
                if (w > width) {
                    w = width;
                }
                if (h > height) {
                    h = height;
                }
                *crop_w = w;
                *crop_h = h;
                *crop_x = (width - w) / 2;
                *crop_y = (height - h) / 2;
            }
        }
    }

    if (cfg->enable_horizontal_flip && cfg->horizontal_flip_prob > 0.0f) {
        float p = random_uniform(0.0f, 1.0f);
        if (p < cfg->horizontal_flip_prob) {
            *flip = 1;
        }
    }
}

static void augmentation_apply_internal(const float *image, int width, int height, int channels,
                                         const AugmentationConfig *cfg, Matrix *dest,
                                         int column)
{
    int expected_rows = cfg->channels * cfg->output_height * cfg->output_width;
    if (dest->rows != expected_rows) {
        /* dimensional mismatch; fill with zeros as a fallback */
        for (int row = 0; row < dest->rows; row++) {
            dest->data[row * dest->cols + column] = 0.0f;
        }
        return;
    }

    int crop_x = 0;
    int crop_y = 0;
    int crop_w = width;
    int crop_h = height;
    int flip = 0;
    augmentation_select_crop(cfg, width, height, &crop_x, &crop_y, &crop_w, &crop_h, &flip);

    float brightness = 1.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    if (cfg->enable_color_jitter) {
        brightness = 1.0f + random_uniform(-cfg->jitter_brightness, cfg->jitter_brightness);
        contrast = 1.0f + random_uniform(-cfg->jitter_contrast, cfg->jitter_contrast);
        saturation = 1.0f + random_uniform(-cfg->jitter_saturation, cfg->jitter_saturation);
    }

    const float contrast_anchor = 0.5f;
    int out_h = cfg->output_height;
    int out_w = cfg->output_width;
    int out_c = cfg->channels;

    float scale_y = (float) crop_h / (float) out_h;
    float scale_x = (float) crop_w / (float) out_w;

    for (int oy = 0; oy < out_h; oy++) {
        float src_y = crop_y + ((oy + 0.5f) * scale_y - 0.5f);
        int y0 = (int) floorf(src_y);
        int y1 = y0 + 1;
        float wy = src_y - (float) y0;

        for (int ox = 0; ox < out_w; ox++) {
            float local_x = (ox + 0.5f) * scale_x - 0.5f;
            if (flip) {
                local_x = (float) (crop_w - 1) - local_x;
            }
            float src_x = crop_x + local_x;
            int x0 = (int) floorf(src_x);
            int x1 = x0 + 1;
            float wx = src_x - (float) x0;

            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;

            for (int c = 0; c < channels; c++) {
                float c00 = sample_hwc_pixel(image, width, height, channels, x0, y0, c);
                float c01 = sample_hwc_pixel(image, width, height, channels, x1, y0, c);
                float c10 = sample_hwc_pixel(image, width, height, channels, x0, y1, c);
                float c11 = sample_hwc_pixel(image, width, height, channels, x1, y1, c);

                float top = c00 + (c01 - c00) * wx;
                float bottom = c10 + (c11 - c10) * wx;
                float value = top + (bottom - top) * wy;

                if (c == 0) {
                    r = value;
                } else if (c == 1) {
                    g = value;
                } else if (c == 2) {
                    b = value;
                }
            }

            if (channels == 1) {
                r = clamp_float(r * brightness, 0.0f, 1.0f);
                r = clamp_float((r - contrast_anchor) * contrast + contrast_anchor, 0.0f, 1.0f);
            } else {
                r = clamp_float(r * brightness, 0.0f, 1.0f);
                g = clamp_float(g * brightness, 0.0f, 1.0f);
                b = clamp_float(b * brightness, 0.0f, 1.0f);

                r = clamp_float((r - contrast_anchor) * contrast + contrast_anchor, 0.0f, 1.0f);
                g = clamp_float((g - contrast_anchor) * contrast + contrast_anchor, 0.0f, 1.0f);
                b = clamp_float((b - contrast_anchor) * contrast + contrast_anchor, 0.0f, 1.0f);

                float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                r = clamp_float(gray + (r - gray) * saturation, 0.0f, 1.0f);
                g = clamp_float(gray + (g - gray) * saturation, 0.0f, 1.0f);
                b = clamp_float(gray + (b - gray) * saturation, 0.0f, 1.0f);
            }

            for (int c = 0; c < out_c; c++) {
                float value = 0.0f;
                if (channels == 1) {
                    value = r;
                } else if (c == 0) {
                    value = r;
                } else if (c == 1) {
                    value = g;
                } else if (c == 2) {
                    value = b;
                }

                if (cfg->enable_normalize && c < 4) {
                    float std_val = cfg->std[c];
                    if (std_val == 0.0f) {
                        std_val = 1.0f;
                    }
                    value = (value - cfg->mean[c]) / std_val;
                }

                int dest_row = (c * out_h + oy) * out_w + ox;
                dest->data[dest_row * dest->cols + column] = (nn_float) value;
            }
        }
    }
}

int augmentation_apply_uint8_to_matrix(const unsigned char *src, AugmentInputLayout layout,
                                       int width, int height, int channels,
                                       const AugmentationConfig *cfg, Matrix *dest,
                                       int column)
{
    if (dest == NULL || src == NULL) {
        return 0;
    }

    if (cfg == NULL) {
        /* simple copy without augmentation */
        int total = width * height * channels;
        if (dest->rows != total) {
            return 0;
        }
        for (int row = 0; row < dest->rows; row++) {
            nn_float value = (nn_float) src[row] / 255.0f;
            dest->data[row * dest->cols + column] = value;
        }
        return 1;
    }

    float *image = (float *) malloc((size_t) width * (size_t) height * (size_t) channels * sizeof(float));
    if (image == NULL) {
        return 0;
    }
    convert_to_hwc(src, layout, width, height, channels, image);
    augmentation_apply_internal(image, width, height, channels, cfg, dest, column);
    free(image);
    return 1;
}

static void set_imagenet_norm(AugmentationConfig *cfg)
{
    cfg->enable_normalize = 1;
    cfg->mean[0] = 0.485f;
    cfg->mean[1] = 0.456f;
    cfg->mean[2] = 0.406f;
    cfg->std[0] = 0.229f;
    cfg->std[1] = 0.224f;
    cfg->std[2] = 0.225f;
    for (int i = 3; i < 4; i++) {
        cfg->mean[i] = 0.0f;
        cfg->std[i] = 1.0f;
    }
}

static void reset_config(AugmentationConfig *cfg, int width, int height, int channels)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->output_width = width;
    cfg->output_height = height;
    cfg->channels = channels;
    set_imagenet_norm(cfg);
}

void augmentation_config_imagenet_train(AugmentationConfig *cfg, int width, int height,
                                        int channels)
{
    reset_config(cfg, width, height, channels);
    cfg->enable_random_resized_crop = 1;
    cfg->scale_min = 0.08f;
    cfg->scale_max = 1.0f;
    cfg->ratio_min = 0.75f;
    cfg->ratio_max = 1.3333f;
    cfg->enable_horizontal_flip = 1;
    cfg->horizontal_flip_prob = 0.5f;
    cfg->enable_color_jitter = 1;
    cfg->jitter_brightness = 0.4f;
    cfg->jitter_contrast = 0.4f;
    cfg->jitter_saturation = 0.4f;
}

void augmentation_config_imagenet_eval(AugmentationConfig *cfg, int width, int height,
                                       int channels)
{
    reset_config(cfg, width, height, channels);
    cfg->enable_center_crop = 1;
    cfg->center_crop_fraction = 0.875f;
    cfg->enable_normalize = 1;
}

void augmentation_config_identity(AugmentationConfig *cfg, int width, int height, int channels,
                                  int normalize)
{
    reset_config(cfg, width, height, channels);
    cfg->enable_normalize = normalize ? 1 : 0;
}
