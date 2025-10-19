#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "matrix.h"
#include "net.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DEFAULT_IMAGENET_TRAIN_RECORD "datasets/tiny-imagenet-200/train"
#define DEFAULT_IMAGENET_VAL_RECORD "datasets/tiny-imagenet-200/val"
#define DEFAULT_IMAGENET_TEST_RECORD "datasets/tiny-imagenet-200/test"

#define IMAGENET_HEIGHT 64
#define IMAGENET_WIDTH 64
#define IMAGENET_CHANNELS 3
#define IMAGENET_CLASSES 200

#define IMAGENET_RECORD_MAGIC 0x4E4E5243u /* 'NNRC' */
#define IMAGENET_RECORD_VERSION 1u

#define IMAGENET_IMAGE_SIZE (IMAGENET_CHANNELS * IMAGENET_HEIGHT * IMAGENET_WIDTH)
#define IMAGENET_RECORD_BYTES (sizeof(uint16_t) + IMAGENET_IMAGE_SIZE)

typedef struct {
    const char *train_record;
    const char *val_record;
    const char *test_record;
    int train_limit;
    int val_limit;
    int test_limit;
    int epochs;
    int batch_size;
    nn_float learning_rate;
    nn_float weight_decay;
    BackendKind backend;
    unsigned int seed;
    int log_steps;
    int log_memory;
} ExampleConfig;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t samples;
} ImagenetRecordHeader;

typedef struct {
    char *path;
    FILE *file;
    int samples;
    long data_offset;
    unsigned char *buffer;
} ImagenetRecord;

typedef struct {
    int channels;
    int height;
    int width;
} FeatureShape;

typedef struct {
    char **wnids;
    int count;
} TinyClassMap;

typedef struct {
    char **paths;
    int *labels;
    int sample_count;
    int capacity;
    int has_labels;
} TinyImageDataset;

typedef enum {
    DATA_SOURCE_NONE = 0,
    DATA_SOURCE_RECORD,
    DATA_SOURCE_TINY_DIR
} DataSourceKind;

typedef enum {
    TINY_SPLIT_TRAIN,
    TINY_SPLIT_VAL,
    TINY_SPLIT_TEST
} TinySplitKind;

static const char *tiny_split_name(TinySplitKind split)
{
    switch (split) {
    case TINY_SPLIT_TRAIN:
        return "train";
    case TINY_SPLIT_VAL:
        return "val";
    case TINY_SPLIT_TEST:
        return "test";
    default:
        return "unknown";
    }
}

typedef struct {
    DataSourceKind kind;
    int sample_count;
    int has_labels;
    union {
        ImagenetRecord record;
        TinyImageDataset tiny;
    } impl;
} DataSource;

static void imagenet_record_close(ImagenetRecord *record);
static int imagenet_record_open(ImagenetRecord *record, const char *path, int limit);
static int imagenet_record_read_column(ImagenetRecord *record, int index, Matrix *input,
                                       Matrix *target, int column);

static const char *datasource_kind_name(DataSourceKind kind)
{
    switch (kind) {
    case DATA_SOURCE_RECORD:
        return "record";
    case DATA_SOURCE_TINY_DIR:
        return "tiny-dir";
    default:
        return "unknown";
    }
}

static void destroy_layer(Layer *layer)
{
    if (layer == NULL) {
        return;
    }
    layer->destroy(layer);
    free(layer);
}

static char *string_duplicate(const char *src)
{
    size_t length = strlen(src);
    char *copy = (char *) malloc(length + 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, src, length + 1);
    return copy;
}

static int path_is_directory(const char *path)
{
    struct stat st;
    if (path == NULL) {
        return 0;
    }
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode);
}

static int path_is_regular_file(const char *path)
{
    struct stat st;
    if (path == NULL) {
        return 0;
    }
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISREG(st.st_mode);
}

static int join_path(char *buffer, size_t buffer_size, const char *a, const char *b)
{
    if (buffer == NULL || buffer_size == 0 || a == NULL || b == NULL) {
        return 0;
    }
    size_t len_a = strlen(a);
    const char *sep = "";
    if (len_a > 0 && a[len_a - 1] != '/') {
        sep = "/";
    }
    int written = snprintf(buffer, buffer_size, "%s%s%s", a, sep, b);
    return written >= 0 && (size_t) written < buffer_size;
}

static char *parent_directory(const char *path)
{
    if (path == NULL) {
        return NULL;
    }
    char *copy = string_duplicate(path);
    if (copy == NULL) {
        return NULL;
    }
    size_t len = strlen(copy);
    while (len > 0 && copy[len - 1] == '/') {
        copy[--len] = '\0';
    }
    char *slash = strrchr(copy, '/');
    if (slash == NULL) {
        return copy;
    }
    if (slash == copy) {
        slash[1] = '\0';
    } else {
        *slash = '\0';
    }
    return copy;
}

static void tiny_class_map_free(TinyClassMap *map)
{
    if (map == NULL) {
        return;
    }
    for (int i = 0; i < map->count; i++) {
        free(map->wnids[i]);
    }
    free(map->wnids);
    map->wnids = NULL;
    map->count = 0;
}

static int tiny_class_map_load(const char *split_dir, TinyClassMap *map)
{
    if (split_dir == NULL || map == NULL) {
        return 0;
    }
    char *root = parent_directory(split_dir);
    if (root == NULL) {
        return 0;
    }
    char wnids_path[PATH_MAX];
    if (!join_path(wnids_path, sizeof(wnids_path), root, "wnids.txt")) {
        free(root);
        return 0;
    }
    free(root);

    FILE *file = fopen(wnids_path, "r");
    if (file == NULL) {
        return 0;
    }

    char **wnids = NULL;
    int count = 0;
    int capacity = 0;
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        char *newline = strchr(line, '\n');
        if (newline != NULL) {
            *newline = '\0';
        }
        if (line[0] == '\0') {
            continue;
        }
        if (count == capacity) {
            int new_capacity = capacity == 0 ? 64 : capacity * 2;
            char **new_array = (char **) realloc(wnids, (size_t) new_capacity * sizeof(char *));
            if (new_array == NULL) {
                fclose(file);
                for (int i = 0; i < count; i++) {
                    free(wnids[i]);
                }
                free(wnids);
                return 0;
            }
            wnids = new_array;
            capacity = new_capacity;
        }
        wnids[count] = string_duplicate(line);
        if (wnids[count] == NULL) {
            fclose(file);
            for (int i = 0; i < count; i++) {
                free(wnids[i]);
            }
            free(wnids);
            return 0;
        }
        count++;
    }
    fclose(file);

    map->wnids = wnids;
    map->count = count;
    return count > 0;
}

static int tiny_class_map_index(const TinyClassMap *map, const char *wnid)
{
    if (map == NULL || wnid == NULL) {
        return -1;
    }
    for (int i = 0; i < map->count; i++) {
        if (strcmp(map->wnids[i], wnid) == 0) {
            return i;
        }
    }
    return -1;
}

static void tiny_dataset_init(TinyImageDataset *dataset, int has_labels)
{
    dataset->paths = NULL;
    dataset->labels = NULL;
    dataset->sample_count = 0;
    dataset->capacity = 0;
    dataset->has_labels = has_labels;
}

static int tiny_dataset_reserve(TinyImageDataset *dataset, int new_capacity)
{
    if (new_capacity <= dataset->capacity) {
        return 1;
    }
    char **new_paths =
        (char **) realloc(dataset->paths, (size_t) new_capacity * sizeof(char *));
    if (new_paths == NULL) {
        return 0;
    }
    dataset->paths = new_paths;
    if (dataset->has_labels) {
        int *new_labels =
            (int *) realloc(dataset->labels, (size_t) new_capacity * sizeof(int));
        if (new_labels == NULL) {
            return 0;
        }
        dataset->labels = new_labels;
    }
    dataset->capacity = new_capacity;
    return 1;
}

static int tiny_dataset_add_sample(TinyImageDataset *dataset, const char *path, int label)
{
    if (dataset->sample_count == dataset->capacity) {
        int new_capacity = dataset->capacity == 0 ? 256 : dataset->capacity * 2;
        if (!tiny_dataset_reserve(dataset, new_capacity)) {
            return 0;
        }
    }
    dataset->paths[dataset->sample_count] = string_duplicate(path);
    if (dataset->paths[dataset->sample_count] == NULL) {
        return 0;
    }
    if (dataset->has_labels) {
        dataset->labels[dataset->sample_count] = label;
    }
    dataset->sample_count++;
    return 1;
}

static void tiny_dataset_free(TinyImageDataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    for (int i = 0; i < dataset->sample_count; i++) {
        free(dataset->paths[i]);
    }
    free(dataset->paths);
    free(dataset->labels);
    dataset->paths = NULL;
    dataset->labels = NULL;
    dataset->sample_count = 0;
    dataset->capacity = 0;
    dataset->has_labels = 0;
}

static int has_extension(const char *name, const char *ext)
{
    size_t name_len = strlen(name);
    size_t ext_len = strlen(ext);
    if (name_len < ext_len) {
        return 0;
    }
    return strcasecmp(name + name_len - ext_len, ext) == 0;
}

static int tiny_dataset_load_train(const char *train_dir, int limit, TinyImageDataset *dataset)
{
    TinyClassMap map = {0};
    if (!tiny_class_map_load(train_dir, &map)) {
        return 0;
    }

    tiny_dataset_init(dataset, 1);

    DIR *root = opendir(train_dir);
    if (root == NULL) {
        tiny_class_map_free(&map);
        return 0;
    }

    struct dirent *entry = NULL;
    while ((entry = readdir(root)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char class_path[PATH_MAX];
        if (!join_path(class_path, sizeof(class_path), train_dir, entry->d_name)) {
            continue;
        }
        if (!path_is_directory(class_path)) {
            continue;
        }

        int label = tiny_class_map_index(&map, entry->d_name);
        if (label < 0) {
            continue;
        }

        char images_path[PATH_MAX];
        if (!join_path(images_path, sizeof(images_path), class_path, "images")) {
            continue;
        }
        DIR *images_dir = opendir(images_path);
        if (images_dir == NULL) {
            continue;
        }
        struct dirent *image_entry = NULL;
        while ((image_entry = readdir(images_dir)) != NULL) {
            if (strcmp(image_entry->d_name, ".") == 0 ||
                strcmp(image_entry->d_name, "..") == 0) {
                continue;
            }
            if (!has_extension(image_entry->d_name, ".JPEG") &&
                !has_extension(image_entry->d_name, ".jpeg") &&
                !has_extension(image_entry->d_name, ".jpg") &&
                !has_extension(image_entry->d_name, ".JPG")) {
                continue;
            }
            char image_path[PATH_MAX];
            if (!join_path(image_path, sizeof(image_path), images_path, image_entry->d_name)) {
                continue;
            }
            if (!tiny_dataset_add_sample(dataset, image_path, label)) {
                closedir(images_dir);
                closedir(root);
                tiny_class_map_free(&map);
                return 0;
            }
            if (limit > 0 && dataset->sample_count >= limit) {
                break;
            }
        }
        closedir(images_dir);
        if (limit > 0 && dataset->sample_count >= limit) {
            break;
        }
    }
    closedir(root);
    tiny_class_map_free(&map);
    return dataset->sample_count > 0;
}

static int tiny_dataset_load_val(const char *val_dir, int limit, TinyImageDataset *dataset)
{
    TinyClassMap map = {0};
    if (!tiny_class_map_load(val_dir, &map)) {
        return 0;
    }

    tiny_dataset_init(dataset, 1);

    char annotations_path[PATH_MAX];
    if (!join_path(annotations_path, sizeof(annotations_path), val_dir, "val_annotations.txt")) {
        tiny_class_map_free(&map);
        return 0;
    }
    FILE *ann = fopen(annotations_path, "r");
    if (ann == NULL) {
        tiny_class_map_free(&map);
        return 0;
    }

    char images_dir[PATH_MAX];
    if (!join_path(images_dir, sizeof(images_dir), val_dir, "images")) {
        fclose(ann);
        tiny_class_map_free(&map);
        return 0;
    }

    char line[512];
    while (fgets(line, sizeof(line), ann) != NULL) {
        char image_name[256];
        char wnid[64];
        if (sscanf(line, "%255s %63s", image_name, wnid) != 2) {
            continue;
        }
        int label = tiny_class_map_index(&map, wnid);
        if (label < 0) {
            continue;
        }
        char image_path[PATH_MAX];
        if (!join_path(image_path, sizeof(image_path), images_dir, image_name)) {
            continue;
        }
        if (!tiny_dataset_add_sample(dataset, image_path, label)) {
            fclose(ann);
            tiny_class_map_free(&map);
            return 0;
        }
        if (limit > 0 && dataset->sample_count >= limit) {
            break;
        }
    }
    fclose(ann);
    tiny_class_map_free(&map);
    return dataset->sample_count > 0;
}

static int tiny_dataset_load_test(const char *test_dir, int limit, TinyImageDataset *dataset)
{
    tiny_dataset_init(dataset, 0);

    char images_dir[PATH_MAX];
    if (!join_path(images_dir, sizeof(images_dir), test_dir, "images")) {
        return 0;
    }
    DIR *dir = opendir(images_dir);
    if (dir == NULL) {
        return 0;
    }
    struct dirent *entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!has_extension(entry->d_name, ".JPEG") && !has_extension(entry->d_name, ".jpeg") &&
            !has_extension(entry->d_name, ".jpg") && !has_extension(entry->d_name, ".JPG")) {
            continue;
        }
        char image_path[PATH_MAX];
        if (!join_path(image_path, sizeof(image_path), images_dir, entry->d_name)) {
            continue;
        }
        if (!tiny_dataset_add_sample(dataset, image_path, -1)) {
            closedir(dir);
            return 0;
        }
        if (limit > 0 && dataset->sample_count >= limit) {
            break;
        }
    }
    closedir(dir);
    return dataset->sample_count > 0;
}

static int tiny_dataset_fill_column(const TinyImageDataset *dataset, int index, Matrix *input,
                                    Matrix *target, int column)
{
    if (dataset == NULL || input == NULL || target == NULL) {
        return 0;
    }
    if (index < 0 || index >= dataset->sample_count) {
        return 0;
    }
    const char *path = dataset->paths[index];

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *pixels =
        stbi_load(path, &width, &height, &channels, IMAGENET_CHANNELS);
    if (pixels == NULL) {
        fprintf(stderr, "Failed to load image %s: %s\n", path, stbi_failure_reason());
        return 0;
    }

    if (width != IMAGENET_WIDTH || height != IMAGENET_HEIGHT) {
        fprintf(stderr,
                "Error: unexpected image size %dx%d for %s (expected %dx%d).\n",
                width, height, path, IMAGENET_WIDTH, IMAGENET_HEIGHT);
        stbi_image_free(pixels);
        return 0;
    }

    int input_stride = input->cols;
    int target_stride = target->cols;

    for (int c = 0; c < IMAGENET_CHANNELS; c++) {
        for (int y = 0; y < IMAGENET_HEIGHT; y++) {
            for (int x = 0; x < IMAGENET_WIDTH; x++) {
                int src_index = (y * IMAGENET_WIDTH + x) * IMAGENET_CHANNELS + c;
                nn_float value = (nn_float) pixels[src_index] / 255.0f;
                int dest_row = (c * IMAGENET_HEIGHT + y) * IMAGENET_WIDTH + x;
                input->data[dest_row * input_stride + column] = value;
            }
        }
    }

    stbi_image_free(pixels);

    for (int row = 0; row < target->rows; row++) {
        target->data[row * target_stride + column] = 0.0f;
    }

    if (dataset->has_labels) {
        int label = dataset->labels[index];
        if (label >= 0 && label < target->rows) {
            target->data[label * target_stride + column] = 1.0f;
        }
    }

    return 1;
}

static int datasource_init(DataSource *source, const char *path, int limit, TinySplitKind split)
{
    if (source == NULL || path == NULL) {
        return 0;
    }
    printf("[data] Loading %s split from %s (limit=%d) ...\n", tiny_split_name(split), path,
           limit);
    fflush(stdout);
    source->kind = DATA_SOURCE_NONE;
    source->sample_count = 0;
    source->has_labels = 0;

    if (path_is_regular_file(path)) {
        printf("[data] Detected binary record file.\n");
        fflush(stdout);
        if (!imagenet_record_open(&source->impl.record, path, limit)) {
            return 0;
        }
        source->kind = DATA_SOURCE_RECORD;
        source->sample_count = source->impl.record.samples;
        source->has_labels = 1;
        return 1;
    }

    if (!path_is_directory(path)) {
        fprintf(stderr, "Path '%s' is neither a file nor a directory.\n", path);
        return 0;
    }

    printf("[data] Scanning Tiny ImageNet directory (split=%s).\n", tiny_split_name(split));
    fflush(stdout);
    int ok = 0;
    if (split == TINY_SPLIT_TRAIN) {
        ok = tiny_dataset_load_train(path, limit, &source->impl.tiny);
    } else if (split == TINY_SPLIT_VAL) {
        ok = tiny_dataset_load_val(path, limit, &source->impl.tiny);
    } else {
        ok = tiny_dataset_load_test(path, limit, &source->impl.tiny);
    }

    if (!ok) {
        return 0;
    }

    source->kind = DATA_SOURCE_TINY_DIR;
    source->sample_count = source->impl.tiny.sample_count;
    source->has_labels = source->impl.tiny.has_labels;
    return 1;
}

static void datasource_close(DataSource *source)
{
    if (source == NULL) {
        return;
    }
    if (source->kind == DATA_SOURCE_RECORD) {
        imagenet_record_close(&source->impl.record);
    } else if (source->kind == DATA_SOURCE_TINY_DIR) {
        tiny_dataset_free(&source->impl.tiny);
    }
    source->kind = DATA_SOURCE_NONE;
    source->sample_count = 0;
    source->has_labels = 0;
}

static int datasource_sample_count(const DataSource *source)
{
    if (source == NULL) {
        return 0;
    }
    return source->sample_count;
}

static int datasource_fill_column(DataSource *source, int index, Matrix *input, Matrix *target,
                                  int column)
{
    if (source == NULL) {
        return 0;
    }
    if (source->kind == DATA_SOURCE_RECORD) {
        return imagenet_record_read_column(&source->impl.record, index, input, target, column);
    }
    if (source->kind == DATA_SOURCE_TINY_DIR) {
        return tiny_dataset_fill_column(&source->impl.tiny, index, input, target, column);
    }
    return 0;
}

static void imagenet_record_close(ImagenetRecord *record)
{
    if (record == NULL) {
        return;
    }
    if (record->file != NULL) {
        fclose(record->file);
    }
    free(record->path);
    free(record->buffer);
    memset(record, 0, sizeof(*record));
}

static int imagenet_record_open(ImagenetRecord *record, const char *path, int limit)
{
    memset(record, 0, sizeof(*record));
    if (path == NULL) {
        return 0;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    ImagenetRecordHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return 0;
    }

    if (header.magic != IMAGENET_RECORD_MAGIC ||
        header.version != IMAGENET_RECORD_VERSION) {
        fclose(file);
        return 0;
    }

    int total = (int) header.samples;
    if (total <= 0) {
        fclose(file);
        return 0;
    }

    if (limit > 0 && limit < total) {
        total = limit;
    }

    unsigned char *buffer = (unsigned char *) malloc(IMAGENET_IMAGE_SIZE);
    if (buffer == NULL) {
        fclose(file);
        return 0;
    }

    record->path = string_duplicate(path);
    if (record->path == NULL) {
        free(buffer);
        fclose(file);
        return 0;
    }

    record->file = file;
    record->samples = total;
    record->data_offset = ftell(file);
    record->buffer = buffer;
    return 1;
}

static int parse_backend(const char *value, BackendKind *backend)
{
    if (strcmp(value, "cpu") == 0) {
        *backend = BACKEND_CPU;
        return 1;
    }
    if (strcmp(value, "gpu") == 0) {
#ifdef USE_CUDA
        *backend = BACKEND_GPU;
        return 1;
#else
        fprintf(stderr, "GPU backend requested but CUDA support is not enabled.\n");
        return 0;
#endif
    }
    fprintf(stderr, "Unrecognized backend '%s'. Use 'cpu' or 'gpu'.\n", value);
    return 0;
}

static void print_usage(const char *prog)
{
    fprintf(
        stderr,
        "Usage: %s [options]\n\n"
        "Arguments:\n"
        "  --train-record PATH   Tiny ImageNet training record or directory (default: %s)\n"
        "  --val-record PATH     Tiny ImageNet validation record or directory (default: %s)\n"
        "  --test-record PATH    Tiny ImageNet test record or directory (default: %s)\n"
        "  --train-limit N       Limit number of training samples (default: all)\n"
        "  --val-limit N         Limit number of validation samples (default: all)\n"
        "  --test-limit N        Limit number of test samples (default: all)\n"
        "  --epochs N            Training epochs (default: 90)\n"
        "  --batch-size N        Mini-batch size (default: 256)\n"
        "  --lr F                Learning rate (default: 0.001)\n"
        "  --weight-decay F      AdamW weight decay (default: 0.0005)\n"
        "  --backend cpu|gpu     Execution backend (default: gpu if available, else "
        "cpu)\n"
        "  --seed N              RNG seed (default: 42)\n"
        "  --log-steps N         Batch logging interval (default: 100, 0 disables)\n"
        "  --log-memory          Log GPU memory usage each step (default: off)\n\n"
        "Record format: little-endian header (magic 0x%08X, version %u, sample "
        "count),\n"
        "followed by samples stored as uint16 label then %d bytes of interleaved RGB "
        "uint8 data\n"
        "(layout NCHW). Provide either prebuilt .rec files or the extracted Tiny ImageNet\n"
        "directory.\n",
        prog, DEFAULT_IMAGENET_TRAIN_RECORD, DEFAULT_IMAGENET_VAL_RECORD,
        DEFAULT_IMAGENET_TEST_RECORD, IMAGENET_RECORD_MAGIC, IMAGENET_RECORD_VERSION,
        IMAGENET_IMAGE_SIZE);
}

static int parse_arguments(int argc, char **argv, ExampleConfig *config)
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--train-record") == 0 && i + 1 < argc) {
            config->train_record = argv[++i];
        } else if (strcmp(arg, "--val-record") == 0 && i + 1 < argc) {
            config->val_record = argv[++i];
        } else if (strcmp(arg, "--test-record") == 0 && i + 1 < argc) {
            config->test_record = argv[++i];
        } else if (strcmp(arg, "--train-limit") == 0 && i + 1 < argc) {
            config->train_limit = atoi(argv[++i]);
        } else if (strcmp(arg, "--val-limit") == 0 && i + 1 < argc) {
            config->val_limit = atoi(argv[++i]);
        } else if (strcmp(arg, "--test-limit") == 0 && i + 1 < argc) {
            config->test_limit = atoi(argv[++i]);
        } else if (strcmp(arg, "--epochs") == 0 && i + 1 < argc) {
            config->epochs = atoi(argv[++i]);
        } else if (strcmp(arg, "--batch-size") == 0 && i + 1 < argc) {
            config->batch_size = atoi(argv[++i]);
        } else if (strcmp(arg, "--lr") == 0 && i + 1 < argc) {
            config->learning_rate = (nn_float) atof(argv[++i]);
        } else if (strcmp(arg, "--weight-decay") == 0 && i + 1 < argc) {
            config->weight_decay = (nn_float) atof(argv[++i]);
        } else if (strcmp(arg, "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend(argv[++i], &config->backend)) {
                return 0;
            }
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            config->seed = (unsigned int) strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--log-steps") == 0 && i + 1 < argc) {
            config->log_steps = atoi(argv[++i]);
        } else if (strcmp(arg, "--log-memory") == 0) {
            config->log_memory = 1;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return 0;
        }
    }
    return 1;
}

static void shuffle_indices(int *indices, int count)
{
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static int imagenet_record_read_column(ImagenetRecord *record, int index, Matrix *input,
                                       Matrix *target, int column)
{
    if (record == NULL || record->file == NULL) {
        return 0;
    }
    if (index < 0 || index >= record->samples) {
        return 0;
    }

    long offset = record->data_offset + (long) index * IMAGENET_RECORD_BYTES;
    if (fseek(record->file, offset, SEEK_SET) != 0) {
        return 0;
    }

    uint16_t label_u16;
    if (fread(&label_u16, sizeof(uint16_t), 1, record->file) != 1) {
        return 0;
    }

    if (fread(record->buffer, 1, IMAGENET_IMAGE_SIZE, record->file) !=
        IMAGENET_IMAGE_SIZE) {
        return 0;
    }

    if (label_u16 >= IMAGENET_CLASSES) {
        return 0;
    }

    for (int row = 0; row < input->rows; row++) {
        input->data[row * input->cols + column] =
            (nn_float) record->buffer[row] / 255.0f;
    }

    for (int row = 0; row < target->rows; row++) {
        target->data[row * target->cols + column] = 0.0f;
    }
    target->data[label_u16 * target->cols + column] = 1.0f;
    return 1;
}

static int count_correct_predictions(const Matrix *predictions, const Matrix *targets,
                                     int columns)
{
    int correct = 0;
    for (int col = 0; col < columns; col++) {
        int pred_label = 0;
        nn_float best = predictions->data[col];
        for (int row = 1; row < predictions->rows; row++) {
            nn_float value = predictions->data[row * predictions->cols + col];
            if (value > best) {
                best = value;
                pred_label = row;
            }
        }
        for (int row = 0; row < targets->rows; row++) {
            if (targets->data[row * targets->cols + col] > 0.5f) {
                if (row == pred_label) {
                    correct++;
                }
                break;
            }
        }
    }
    return correct;
}

static int conv_output_dim(int input, int kernel, int stride, int padding)
{
    return (input + 2 * padding - kernel) / stride + 1;
}

static int add_conv_relu(Network *nn, BackendKind backend, FeatureShape *shape,
                         int out_channels, int kernel_h, int kernel_w, int stride_h,
                         int stride_w, int pad_h, int pad_w)
{
    int out_h = conv_output_dim(shape->height, kernel_h, stride_h, pad_h);
    int out_w = conv_output_dim(shape->width, kernel_w, stride_w, pad_w);
    if (out_h <= 0 || out_w <= 0) {
        return 0;
    }

    Layer *conv = layer_conv2d_create_backend(
        backend, shape->channels, out_channels, shape->height, shape->width, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w);
    Layer *relu = layer_activation_from_kind_backend(backend, ACT_RELU);
    if (conv == NULL || relu == NULL) {
        destroy_layer(conv);
        destroy_layer(relu);
        return 0;
    }

    nn_add_layer(nn, conv);
    nn_add_layer(nn, relu);
    shape->channels = out_channels;
    shape->height = out_h;
    shape->width = out_w;
    return 1;
}

static int add_stage_transition(Network *nn, BackendKind backend, FeatureShape *shape,
                                int out_channels)
{
    return add_conv_relu(nn, backend, shape, out_channels, 1, 1, 2, 2, 0, 0);
}

static int add_bottleneck_block(Network *nn, BackendKind backend, FeatureShape *shape,
                                int bottleneck_channels)
{
    if (shape->channels <= 0) {
        return 0;
    }

    SkipConnection skip = skip_connection_create();
    Layer *conv1 =
        layer_conv2d_create_backend(backend, shape->channels, bottleneck_channels,
                                    shape->height, shape->width, 1, 1, 1, 1, 0, 0);
    Layer *relu1 = layer_activation_from_kind_backend(backend, ACT_RELU);
    Layer *conv2 =
        layer_conv2d_create_backend(backend, bottleneck_channels, bottleneck_channels,
                                    shape->height, shape->width, 3, 3, 1, 1, 1, 1);
    Layer *relu2 = layer_activation_from_kind_backend(backend, ACT_RELU);
    Layer *conv3 =
        layer_conv2d_create_backend(backend, bottleneck_channels, shape->channels,
                                    shape->height, shape->width, 1, 1, 1, 1, 0, 0);
    Layer *relu_out = layer_activation_from_kind_backend(backend, ACT_RELU);

    if (skip.save == NULL || skip.add == NULL || conv1 == NULL || relu1 == NULL ||
        conv2 == NULL || relu2 == NULL || conv3 == NULL || relu_out == NULL) {
        destroy_layer(skip.save);
        destroy_layer(skip.add);
        destroy_layer(conv1);
        destroy_layer(relu1);
        destroy_layer(conv2);
        destroy_layer(relu2);
        destroy_layer(conv3);
        destroy_layer(relu_out);
        return 0;
    }

    nn_add_layer(nn, skip.save);
    nn_add_layer(nn, conv1);
    nn_add_layer(nn, relu1);
    nn_add_layer(nn, conv2);
    nn_add_layer(nn, relu2);
    nn_add_layer(nn, conv3);
    nn_add_layer(nn, skip.add);
    nn_add_layer(nn, relu_out);
    return 1;
}

static Network *build_resnet50_like(BackendKind backend)
{
    const int input_size = IMAGENET_IMAGE_SIZE;
    Network *nn = nn_create_empty(input_size, IMAGENET_CLASSES, backend);
    if (nn == NULL) {
        return NULL;
    }

    FeatureShape shape = {.channels = IMAGENET_CHANNELS,
                          .height = IMAGENET_HEIGHT,
                          .width = IMAGENET_WIDTH};

    if (!add_conv_relu(nn, backend, &shape, 64, 7, 7, 2, 2, 3, 3)) {
        nn_free(nn);
        return NULL;
    }
    if (!add_conv_relu(nn, backend, &shape, 64, 3, 3, 2, 2, 1, 1)) {
        nn_free(nn);
        return NULL;
    }

    for (int i = 0; i < 3; i++) {
        if (!add_bottleneck_block(nn, backend, &shape, 64)) {
            nn_free(nn);
            return NULL;
        }
    }

    if (!add_stage_transition(nn, backend, &shape, 128)) {
        nn_free(nn);
        return NULL;
    }
    for (int i = 0; i < 4; i++) {
        if (!add_bottleneck_block(nn, backend, &shape, 128)) {
            nn_free(nn);
            return NULL;
        }
    }

    if (!add_stage_transition(nn, backend, &shape, 256)) {
        nn_free(nn);
        return NULL;
    }
    for (int i = 0; i < 6; i++) {
        if (!add_bottleneck_block(nn, backend, &shape, 256)) {
            nn_free(nn);
            return NULL;
        }
    }

    if (!add_stage_transition(nn, backend, &shape, 512)) {
        nn_free(nn);
        return NULL;
    }
    for (int i = 0; i < 3; i++) {
        if (!add_bottleneck_block(nn, backend, &shape, 512)) {
            nn_free(nn);
            return NULL;
        }
    }

    Layer *dropout = layer_dropout_create(0.3f);
    if (dropout == NULL) {
        nn_free(nn);
        return NULL;
    }
    nn_add_layer(nn, dropout);

    int flattened = shape.channels * shape.height * shape.width;
    Layer *dense1 = layer_dense_create_backend(backend, flattened, 2048);
    if (dense1 == NULL) {
        nn_free(nn);
        return NULL;
    }
    nn_add_layer(nn, dense1);

    Layer *relu = layer_activation_from_kind_backend(backend, ACT_RELU);
    if (relu == NULL) {
        nn_free(nn);
        return NULL;
    }
    nn_add_layer(nn, relu);

    Layer *dense2 = layer_dense_create_backend(backend, 2048, IMAGENET_CLASSES);
    if (dense2 == NULL) {
        nn_free(nn);
        return NULL;
    }
    nn_add_layer(nn, dense2);

    Layer *softmax = layer_softmax_create();
    if (softmax == NULL) {
        nn_free(nn);
        return NULL;
    }
    nn_add_layer(nn, softmax);

    return nn;
}

static void evaluate_source(Network *nn, DataSource *source, const char *label, int batch_size,
                            Matrix *input, Matrix *target, Matrix *prediction,
                            double *loss_out, double *acc_out)
{
    int total = datasource_sample_count(source);
    if (total <= 0) {
        if (loss_out != NULL) {
            *loss_out = 0.0;
        }
        if (acc_out != NULL) {
            *acc_out = 0.0;
        }
        return;
    }

    int saved_input_cols = input->cols;
    int saved_target_cols = target->cols;
    int saved_pred_cols = prediction->cols;

    double loss_sum = 0.0;
    int correct = 0;

    if (label != NULL) {
        printf("[eval] %s: %d samples (batch=%d, labels=%s)\n", label, total, batch_size,
               source->has_labels ? "yes" : "no");
        fflush(stdout);
    }

    for (int start = 0; start < total; start += batch_size) {
        int current = total - start;
        if (current > batch_size) {
            current = batch_size;
        }

        input->cols = current;
        target->cols = current;
        prediction->cols = current;
        matrix_zero(target);

        int ok = 1;
        for (int col = 0; col < current; col++) {
            if (!datasource_fill_column(source, start + col, input, target, col)) {
                fprintf(stderr, "Failed to load sample %d from source.\n", start + col);
                ok = 0;
                break;
            }
        }
        if (!ok) {
            break;
        }

        Matrix *pred = nn_forward(nn, input);
        if (pred == NULL) {
            fprintf(stderr, "Forward pass failed during evaluation.\n");
            break;
        }
        matrix_copy_into(prediction, pred);
        matrix_free(pred);

        if (source->has_labels) {
            for (int col = 0; col < current; col++) {
                for (int row = 0; row < target->rows; row++) {
                    if (target->data[row * target->cols + col] > 0.5f) {
                        nn_float prob = prediction->data[row * prediction->cols + col];
                        if (prob < 1e-7f) {
                            prob = 1e-7f;
                        }
                        loss_sum += -log((double) prob);
                        break;
                    }
                }
            }
            correct += count_correct_predictions(prediction, target, current);
        }

    }

    input->cols = saved_input_cols;
    target->cols = saved_target_cols;
    prediction->cols = saved_pred_cols;

    if (loss_out != NULL) {
        *loss_out = source->has_labels ? loss_sum / (double) total : 0.0;
    }
    if (acc_out != NULL) {
        *acc_out = source->has_labels ? (double) correct / (double) total * 100.0 : 0.0;
    }
}

int main(int argc, char **argv)
{
    ExampleConfig config = {
        .train_record = DEFAULT_IMAGENET_TRAIN_RECORD,
        .val_record = DEFAULT_IMAGENET_VAL_RECORD,
        .test_record = DEFAULT_IMAGENET_TEST_RECORD,
        .train_limit = -1,
        .val_limit = -1,
        .test_limit = -1,
        .epochs = 90,
        .batch_size = 256,
        .learning_rate = 0.001f,
        .weight_decay = 0.0005f,
        .backend =
#ifdef USE_CUDA
            BACKEND_GPU,
#else
            BACKEND_CPU,
#endif
        .seed = 42U,
        .log_steps = 100,
        .log_memory = 0,
    };

    if (!parse_arguments(argc, argv, &config)) {
        print_usage(argv[0]);
        return 1;
    }

    if (config.batch_size <= 0 || config.epochs <= 0) {
        fprintf(stderr, "Batch size and epochs must be positive.\n");
        return 1;
    }
    if (config.learning_rate <= 0.0f) {
        fprintf(stderr, "Learning rate must be positive.\n");
        return 1;
    }
    if (config.log_steps < 0) {
        fprintf(stderr, "log-steps must be non-negative.\n");
        return 1;
    }

    srand(config.seed);

    DataSource train_source = {0};
    if (!datasource_init(&train_source, config.train_record, config.train_limit,
                         TINY_SPLIT_TRAIN)) {
        fprintf(stderr, "Failed to load training data from '%s'.\n", config.train_record);
        return 1;
    }
    if (!train_source.has_labels) {
        fprintf(stderr, "Training data must include labels.\n");
        datasource_close(&train_source);
        return 1;
    }
    fflush(stdout);

    DataSource val_source = {0};
    int have_val = 0;
    if (config.val_record != NULL && strlen(config.val_record) > 0) {
        if (!datasource_init(&val_source, config.val_record, config.val_limit,
                             TINY_SPLIT_VAL)) {
            fprintf(stderr, "Failed to load validation data from '%s'.\n",
                    config.val_record);
            datasource_close(&train_source);
            return 1;
        }
        have_val = datasource_sample_count(&val_source) > 0 && val_source.has_labels;
        printf("[data] Loaded %d samples (%s) from %s [%s]\n",
               datasource_sample_count(&val_source),
               val_source.has_labels ? "labeled" : "unlabeled", config.val_record,
               datasource_kind_name(val_source.kind));
        fflush(stdout);
    }

    DataSource test_source = {0};
    if (config.test_record != NULL && strlen(config.test_record) > 0) {
        if (!datasource_init(&test_source, config.test_record, config.test_limit,
                             TINY_SPLIT_TEST)) {
            fprintf(stderr, "Failed to load test data from '%s'.\n", config.test_record);
            datasource_close(&train_source);
            datasource_close(&val_source);
            return 1;
        }
        printf("[data] Loaded %d samples (%s) from %s [%s]\n",
               datasource_sample_count(&test_source),
               test_source.has_labels ? "labeled" : "unlabeled", config.test_record,
               datasource_kind_name(test_source.kind));
        fflush(stdout);
    }

    if (datasource_sample_count(&train_source) <= 0) {
        fprintf(stderr, "Training dataset is empty.\n");
        datasource_close(&train_source);
        datasource_close(&val_source);
        datasource_close(&test_source);
        return 1;
    }

    Network *nn = build_resnet50_like(config.backend);
    if (nn == NULL) {
        fprintf(stderr, "Failed to create ResNet-like network.\n");
        datasource_close(&train_source);
        datasource_close(&val_source);
        datasource_close(&test_source);
        return 1;
    }

    nn_set_optimizer(nn, OPTIMIZER_ADAMW, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    nn_set_seed(nn, config.seed);
    nn_set_log_memory(nn, config.log_memory);

    printf("=== ImageNet ResNet Example ===\n\n");
    printf("Train samples: %d\n", datasource_sample_count(&train_source));
    printf("Val samples:   %d\n", datasource_sample_count(&val_source));
    printf("Test samples:  %d\n\n", datasource_sample_count(&test_source));
    printf("Config:\n");
    printf("  backend:          %s\n", backend_kind_name(config.backend));
    printf("  optimizer:        %s\n", optimizer_kind_name(nn->optimizer.kind));
    printf("  learning_rate:    %.6f\n", config.learning_rate);
    printf("  weight_decay:     %.6f\n", config.weight_decay);
    printf("  epochs:           %d\n", config.epochs);
    printf("  batch_size:       %d\n", config.batch_size);
    printf("  log_steps:        %d\n", config.log_steps);
    printf("  log_memory:       %s\n", config.log_memory ? "yes" : "no");
    printf("  seed:             %u\n", config.seed);
    printf("  train_limit:      %d\n", config.train_limit);
    printf("  val_limit:        %d\n", config.val_limit);
    printf("  test_limit:       %d\n", config.test_limit);
    printf("  train_source:     %s (%s)\n", config.train_record,
           datasource_kind_name(train_source.kind));
    if (config.val_record != NULL && strlen(config.val_record) > 0) {
        printf("  val_source:       %s (%s)\n", config.val_record,
               datasource_kind_name(val_source.kind));
    }
    if (config.test_record != NULL && strlen(config.test_record) > 0) {
        printf("  test_source:      %s (%s)\n", config.test_record,
               datasource_kind_name(test_source.kind));
    }
    printf("\nModel Architecture:\n");
    nn_print_architecture(nn);
    printf("\n");

    if (config.backend == BACKEND_GPU) {
        double flops_per_sample = nn_estimated_flops_per_sample(nn);
        printf("Estimated FLOPs/sample: %.3f GFLOPs\n",
               flops_per_sample / 1e9);
        printf("Estimated FLOPs/step (batch %d): %.3f GFLOPs\n\n", config.batch_size,
               flops_per_sample * (double) config.batch_size / 1e9);
    }

    Matrix *batch_inputs = matrix_create(IMAGENET_IMAGE_SIZE, config.batch_size);
    Matrix *batch_targets = matrix_create(IMAGENET_CLASSES, config.batch_size);
    Matrix *batch_predictions = matrix_create(IMAGENET_CLASSES, config.batch_size);
    if (batch_inputs == NULL || batch_targets == NULL || batch_predictions == NULL) {
        fprintf(stderr, "Failed to allocate batch matrices.\n");
        matrix_free(batch_inputs);
        matrix_free(batch_targets);
        matrix_free(batch_predictions);
        nn_free(nn);
        datasource_close(&train_source);
        datasource_close(&val_source);
        datasource_close(&test_source);
        return 1;
    }

    int train_samples = datasource_sample_count(&train_source);
    int *train_indices = (int *) malloc((size_t) train_samples * sizeof(int));
    if (train_indices == NULL) {
        fprintf(stderr, "Failed to allocate index buffer.\n");
        matrix_free(batch_inputs);
        matrix_free(batch_targets);
        matrix_free(batch_predictions);
        nn_free(nn);
        datasource_close(&train_source);
        datasource_close(&val_source);
        datasource_close(&test_source);
        return 1;
    }
    for (int i = 0; i < train_samples; i++) {
        train_indices[i] = i;
    }

    nn_set_training(nn, 1);

    clock_t train_start_clock = clock();
    long long total_steps = 0;
    double samples_processed = 0.0;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        shuffle_indices(train_indices, train_samples);

        double epoch_loss = 0.0;
        double epoch_grad = 0.0;
        int epoch_correct = 0;
        double epoch_mfu_sum = 0.0;
        size_t last_vram_used_bytes = 0;
        size_t last_vram_total_bytes = 0;

        for (int start = 0; start < train_samples; start += config.batch_size) {
            int end = start + config.batch_size;
            if (end > train_samples) {
                end = train_samples;
            }
            int current_batch = end - start;

            int saved_input_cols = batch_inputs->cols;
            int saved_target_cols = batch_targets->cols;
            int saved_pred_cols = batch_predictions->cols;
            batch_inputs->cols = current_batch;
            batch_targets->cols = current_batch;
            batch_predictions->cols = current_batch;
            matrix_zero(batch_targets);

            int ok = 1;
            for (int col = 0; col < current_batch; col++) {
                int idx = train_indices[start + col];
                if (!datasource_fill_column(&train_source, idx, batch_inputs, batch_targets,
                                            col)) {
                    fprintf(stderr, "Failed to load training sample %d.\n", idx);
                    ok = 0;
                    break;
                }
            }

            if (!ok) {
                batch_inputs->cols = saved_input_cols;
                batch_targets->cols = saved_target_cols;
                batch_predictions->cols = saved_pred_cols;
                free(train_indices);
                matrix_free(batch_inputs);
                matrix_free(batch_targets);
                matrix_free(batch_predictions);
                nn_free(nn);
                datasource_close(&train_source);
                datasource_close(&val_source);
                datasource_close(&test_source);
                return 1;
            }

            nn_zero_gradients(nn);
            TrainStepStats stats =
                nn_train_batch(nn, batch_inputs, batch_targets, current_batch, batch_predictions);
            nn_apply_gradients(nn, config.learning_rate, current_batch);

            epoch_loss += stats.loss * stats.samples;
            epoch_grad += stats.grad_norm * stats.samples;
            epoch_mfu_sum += (double) stats.mfu * stats.samples;
            if (config.backend == BACKEND_GPU) {
                last_vram_used_bytes = stats.vram_used_bytes;
                if (stats.vram_total_bytes > 0) {
                    last_vram_total_bytes = stats.vram_total_bytes;
                }
            }
            int correct = 0;
            if (train_source.has_labels) {
                correct = count_correct_predictions(batch_predictions, batch_targets,
                                                    current_batch);
                epoch_correct += correct;
            }

            total_steps += 1;
            samples_processed += (double) stats.samples;

            if (config.log_steps > 0 &&
                (total_steps == 1 || total_steps % config.log_steps == 0)) {
                clock_t now = clock();
                double elapsed = (double) (now - train_start_clock) / CLOCKS_PER_SEC;
                double samples_per_sec =
                    elapsed > 0.0 ? samples_processed / elapsed : 0.0;
                double batch_acc =
                    train_source.has_labels && current_batch > 0
                        ? (double) correct / (double) current_batch * 100.0
                        : 0.0;
                if (config.backend == BACKEND_GPU) {
                    if (stats.vram_total_bytes > 0) {
                        double vram_total_gb =
                            (double) stats.vram_total_bytes / (1024.0 * 1024.0 * 1024.0);
                        double vram_used_gb =
                            (double) stats.vram_used_bytes / (1024.0 * 1024.0 * 1024.0);
                        double vram_pct = (double) stats.vram_used_bytes /
                                          (double) stats.vram_total_bytes * 100.0;
                        printf(
                            "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                            "VRAM: %.2f/%.2f GB (%.1f%%) | Samples/s: %.2fK\n",
                            total_steps, stats.loss, batch_acc, stats.grad_norm,
                            (double) stats.mfu * 100.0, vram_used_gb, vram_total_gb, vram_pct,
                            samples_per_sec / 1000.0);
                    } else {
                        printf(
                            "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                            "VRAM: n/a | Samples/s: %.2fK\n",
                            total_steps, stats.loss, batch_acc, stats.grad_norm,
                            (double) stats.mfu * 100.0, samples_per_sec / 1000.0);
                    }
                } else {
                    printf(
                        "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | Samples/s: "
                        "%.2fK\n",
                        total_steps, stats.loss, batch_acc, stats.grad_norm,
                        samples_per_sec / 1000.0);
                }
            }

            batch_inputs->cols = saved_input_cols;
            batch_targets->cols = saved_target_cols;
            batch_predictions->cols = saved_pred_cols;
        }

        epoch_loss /= (double) train_samples;
        epoch_grad /= (double) train_samples;
        double epoch_acc = train_source.has_labels
                                ? (double) epoch_correct / (double) train_samples * 100.0
                                : 0.0;
        double epoch_mfu =
            train_samples > 0 ? epoch_mfu_sum / (double) train_samples : 0.0;

        double val_loss = 0.0;
        double val_acc = 0.0;
        if (have_val) {
            nn_set_training(nn, 0);
            evaluate_source(nn, &val_source, "validation", config.batch_size, batch_inputs,
                            batch_targets, batch_predictions, &val_loss, &val_acc);
            nn_set_training(nn, 1);
        }

        if (config.backend == BACKEND_GPU) {
            if (last_vram_total_bytes > 0) {
                double vram_total_gb =
                    (double) last_vram_total_bytes / (1024.0 * 1024.0 * 1024.0);
                double vram_used_gb =
                    (double) last_vram_used_bytes / (1024.0 * 1024.0 * 1024.0);
                double vram_pct = (double) last_vram_used_bytes /
                                  (double) last_vram_total_bytes * 100.0;
                printf(
                    "Epoch %3d | TrainLoss: %.5f | TrainAcc: %.2f%% | Grad: %.5f | MFU: %.2f%% | "
                    "VRAM: %.2f/%.2f GB (%.1f%%)",
                    epoch + 1, epoch_loss, epoch_acc, epoch_grad, epoch_mfu * 100.0,
                    vram_used_gb, vram_total_gb, vram_pct);
            } else {
                printf(
                    "Epoch %3d | TrainLoss: %.5f | TrainAcc: %.2f%% | Grad: %.5f | MFU: %.2f%% | "
                    "VRAM: n/a",
                    epoch + 1, epoch_loss, epoch_acc, epoch_grad, epoch_mfu * 100.0);
            }
        } else {
            printf("Epoch %3d | TrainLoss: %.5f | TrainAcc: %.2f%% | Grad: %.5f", epoch + 1,
                   epoch_loss, epoch_acc, epoch_grad);
        }
        if (have_val) {
            printf(" | ValLoss: %.5f | ValAcc: %.2f%%", val_loss, val_acc);
        }
        printf("\n");
    }

    nn_set_training(nn, 0);

    double final_train_loss = 0.0;
    double final_train_acc = 0.0;
    evaluate_source(nn, &train_source, "train", config.batch_size, batch_inputs, batch_targets,
                    batch_predictions, &final_train_loss, &final_train_acc);

    double final_val_loss = 0.0;
    double final_val_acc = 0.0;
    if (datasource_sample_count(&val_source) > 0) {
        evaluate_source(nn, &val_source, "validation", config.batch_size, batch_inputs,
                        batch_targets, batch_predictions, &final_val_loss, &final_val_acc);
    }

    double final_test_loss = 0.0;
    double final_test_acc = 0.0;
    if (datasource_sample_count(&test_source) > 0) {
        evaluate_source(nn, &test_source, "test", config.batch_size, batch_inputs,
                        batch_targets, batch_predictions, &final_test_loss,
                        &final_test_acc);
    }

    printf("\nFinal Train | Loss: %.5f | Acc: %.2f%%\n", final_train_loss,
           final_train_acc);
    if (datasource_sample_count(&val_source) > 0) {
        if (val_source.has_labels) {
            printf("Final Val   | Loss: %.5f | Acc: %.2f%%\n", final_val_loss,
                   final_val_acc);
        } else {
            printf("Final Val   | Loss: %.5f | Acc: N/A (unlabeled)\n",
                   final_val_loss);
        }
    }
    if (datasource_sample_count(&test_source) > 0) {
        if (test_source.has_labels) {
            printf("Final Test  | Loss: %.5f | Acc: %.2f%%\n", final_test_loss,
                   final_test_acc);
        } else {
            printf("Final Test  | Loss: %.5f | Acc: N/A (unlabeled)\n",
                   final_test_loss);
        }
    }

    free(train_indices);
    matrix_free(batch_inputs);
    matrix_free(batch_targets);
    matrix_free(batch_predictions);
    nn_free(nn);

    datasource_close(&train_source);
    datasource_close(&val_source);
    datasource_close(&test_source);

    return 0;
}
