#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"
#include "matrix.h"
#include "net.h"

#define DEFAULT_MNIST_TRAIN_IMAGES "datasets/mnist/train-images.idx3-ubyte"
#define DEFAULT_MNIST_TRAIN_LABELS "datasets/mnist/train-labels.idx1-ubyte"
#define DEFAULT_MNIST_TEST_IMAGES "datasets/mnist/t10k-images.idx3-ubyte"
#define DEFAULT_MNIST_TEST_LABELS "datasets/mnist/t10k-labels.idx1-ubyte"

typedef struct {
    const char *train_images;
    const char *train_labels;
    const char *test_images;
    const char *test_labels;
    int limit;
    int test_limit;
    int epochs;
    int batch_size;
    double val_split;
    nn_float learning_rate;
    nn_float weight_decay;
    BackendKind backend;
    unsigned int seed;
    int log_steps;
    int use_residual;
    int log_memory;
} ExampleConfig;

static void print_usage(const char *prog)
{
    fprintf(
        stderr,
        "Usage: %s [options]\n\n"
        "Arguments:\n"
        "  --train-images PATH   MNIST training images (default: %s)\n"
        "  --train-labels PATH   MNIST training labels (default: %s)\n"
        "  --test-images PATH    MNIST test images for final evaluation (default: %s)\n"
        "  --test-labels PATH    MNIST test labels for final evaluation (default: %s)\n"
        "  --limit N             Limit number of training samples (default: 60000)\n"
        "  --test-limit N        Limit number of test samples (default: all)\n"
        "  --epochs N            Training epochs (default: 5)\n"
        "  --batch-size N        Mini-batch size (default: 64)\n"
        "  --val-split F         Validation split fraction (default: 0.15)\n"
        "  --lr F                Learning rate (default: 0.001)\n"
        "  --weight-decay F      AdamW weight decay (default: 0.0005)\n"
        "  --backend cpu|gpu     Execution backend (default: cpu)\n"
        "  --seed N              RNG seed (default: 42)\n"
        "  --log-steps N         Batch logging interval (default: 100, 0 disables)\n"
        "  --no-residual          Disable residual fully-connected block\n"
        "  --residual             Enable residual fully-connected block\n",
        prog, DEFAULT_MNIST_TRAIN_IMAGES, DEFAULT_MNIST_TRAIN_LABELS,
        DEFAULT_MNIST_TEST_IMAGES, DEFAULT_MNIST_TEST_LABELS);
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

static int parse_arguments(int argc, char **argv, ExampleConfig *config)
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--train-images") == 0 && i + 1 < argc) {
            config->train_images = argv[++i];
        } else if (strcmp(arg, "--train-labels") == 0 && i + 1 < argc) {
            config->train_labels = argv[++i];
        } else if (strcmp(arg, "--test-images") == 0 && i + 1 < argc) {
            config->test_images = argv[++i];
        } else if (strcmp(arg, "--test-labels") == 0 && i + 1 < argc) {
            config->test_labels = argv[++i];
        } else if (strcmp(arg, "--limit") == 0 && i + 1 < argc) {
            config->limit = atoi(argv[++i]);
        } else if (strcmp(arg, "--test-limit") == 0 && i + 1 < argc) {
            config->test_limit = atoi(argv[++i]);
        } else if (strcmp(arg, "--epochs") == 0 && i + 1 < argc) {
            config->epochs = atoi(argv[++i]);
        } else if (strcmp(arg, "--batch-size") == 0 && i + 1 < argc) {
            config->batch_size = atoi(argv[++i]);
        } else if (strcmp(arg, "--val-split") == 0 && i + 1 < argc) {
            config->val_split = atof(argv[++i]);
        } else if (strcmp(arg, "--lr") == 0 && i + 1 < argc) {
            config->learning_rate = (nn_float) atof(argv[++i]);
        } else if (strcmp(arg, "--weight-decay") == 0 && i + 1 < argc) {
            config->weight_decay = (nn_float) atof(argv[++i]);
        } else if (strcmp(arg, "--log-memory") == 0) {
            config->log_memory = 1;
        } else if (strcmp(arg, "--backend") == 0 && i + 1 < argc) {
            if (!parse_backend(argv[++i], &config->backend)) {
                return 0;
            }
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            config->seed = (unsigned int) strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--log-steps") == 0 && i + 1 < argc) {
            config->log_steps = atoi(argv[++i]);
        } else if (strcmp(arg, "--no-residual") == 0) {
            config->use_residual = 0;
        } else if (strcmp(arg, "--residual") == 0) {
            config->use_residual = 1;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return 0;
        }
    }
    if (config->train_images == NULL || config->train_labels == NULL) {
        fprintf(stderr, "Training image/label paths are required.\n");
        return 0;
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

static int argmax_column(const Matrix *matrix, int col)
{
    int rows = matrix->rows;
    int best_index = 0;
    nn_float best_value = matrix->data[col];
    for (int row = 1; row < rows; row++) {
        nn_float value = matrix->data[row * matrix->cols + col];
        if (value > best_value) {
            best_value = value;
            best_index = row;
        }
    }
    return best_index;
}

static int count_correct_predictions(const Matrix *predictions, const Matrix *targets,
                                     int columns)
{
    int correct = 0;
    for (int col = 0; col < columns; col++) {
        int pred_label = argmax_column(predictions, col);
        int true_label = argmax_column(targets, col);
        if (pred_label == true_label) {
            correct++;
        }
    }
    return correct;
}

static void evaluate_subset(Network *nn, const Dataset *dataset, const int *indices,
                            int count, Matrix *input, Matrix *target, double *loss_out,
                            double *acc_out)
{
    if (count <= 0) {
        if (loss_out != NULL) {
            *loss_out = 0.0;
        }
        if (acc_out != NULL) {
            *acc_out = 0.0;
        }
        return;
    }

    const double epsilon = 1e-7;
    double loss_sum = 0.0;
    int correct = 0;

    for (int i = 0; i < count; i++) {
        dataset_fill_sample(dataset, indices[i], input, target);
        Matrix *pred = nn_forward(nn, input);

        for (int row = 0; row < target->rows; row++) {
            nn_float y = target->data[row];
            if (y > 0.0f) {
                nn_float y_hat = pred->data[row];
                if (y_hat < epsilon) {
                    y_hat = (nn_float) epsilon;
                }
                loss_sum += -log((double) y_hat);
            }
        }

        if (argmax_column(pred, 0) == argmax_column(target, 0)) {
            correct++;
        }
        matrix_free(pred);
    }

    if (loss_out != NULL) {
        *loss_out = loss_sum / (double) count;
    }
    if (acc_out != NULL) {
        *acc_out = (double) correct / (double) count * 100.0;
    }
}

static Network *build_cnn(const Dataset *dataset, ExampleConfig *config)
{
    int image_size = dataset->input_size;
    int output_size = dataset->output_size;
    BackendKind backend = config->backend;
    int side = (int) round(sqrt((double) image_size));
    if (side * side != image_size) {
        fprintf(stderr,
                "Input size %d is not a perfect square; cannot form 2D images.\n",
                image_size);
        return NULL;
    }

    const int in_channels = 1;

    Network *nn = nn_create_empty(image_size, output_size, backend);
    if (nn == NULL) {
        return NULL;
    }

    Layer *conv1 = layer_conv2d_create_backend(backend, in_channels, 8, side, side, 5,
                                               5, 1, 1, 2, 2);
    Layer *relu1 = layer_activation_from_kind_backend(backend, ACT_RELU);
    Layer *conv2 =
        layer_conv2d_create_backend(backend, 8, 16, side, side, 3, 3, 1, 1, 1, 1);
    Layer *relu2 = layer_activation_from_kind_backend(backend, ACT_RELU);
    Layer *dropout = layer_dropout_create(0.25f);
    int dense_input = 16 * side * side;
    Layer *dense1 = layer_dense_create_backend(backend, dense_input, 128);
    Layer *relu3 = layer_activation_from_kind_backend(backend, ACT_RELU);
    Layer *dense_residual = NULL;
    Layer *relu_residual = NULL;
    SkipConnection residual_fc = {0};
    if (config->use_residual) {
        residual_fc = skip_connection_create();
        if (residual_fc.save == NULL || residual_fc.add == NULL) {
            if (residual_fc.save != NULL) {
                residual_fc.save->destroy(residual_fc.save);
                free(residual_fc.save);
            }
            if (residual_fc.add != NULL) {
                residual_fc.add->destroy(residual_fc.add);
                free(residual_fc.add);
            }
            config->use_residual = 0;
        } else {
            dense_residual = layer_dense_create_backend(backend, 128, 128);
            relu_residual = layer_activation_from_kind_backend(backend, ACT_RELU);
            if (dense_residual == NULL || relu_residual == NULL) {
                if (dense_residual != NULL) {
                    dense_residual->destroy(dense_residual);
                    free(dense_residual);
                    dense_residual = NULL;
                }
                if (relu_residual != NULL) {
                    relu_residual->destroy(relu_residual);
                    free(relu_residual);
                    relu_residual = NULL;
                }
                residual_fc.save->destroy(residual_fc.save);
                free(residual_fc.save);
                residual_fc.add->destroy(residual_fc.add);
                free(residual_fc.add);
                residual_fc.save = residual_fc.add = NULL;
                config->use_residual = 0;
            }
        }
    }
    Layer *dense2 = layer_dense_create_backend(backend, 128, output_size);
    Layer *softmax = layer_softmax_create();

    if (conv1 == NULL || relu1 == NULL || conv2 == NULL || relu2 == NULL ||
        dropout == NULL || dense1 == NULL || relu3 == NULL || dense2 == NULL ||
        softmax == NULL ||
        (config->use_residual && (dense_residual == NULL || relu_residual == NULL))) {
        if (conv1 != NULL) {
            conv1->destroy(conv1);
            free(conv1);
        }
        if (relu1 != NULL) {
            relu1->destroy(relu1);
            free(relu1);
        }
        if (conv2 != NULL) {
            conv2->destroy(conv2);
            free(conv2);
        }
        if (relu2 != NULL) {
            relu2->destroy(relu2);
            free(relu2);
        }
        if (dropout != NULL) {
            dropout->destroy(dropout);
            free(dropout);
        }
        if (dense1 != NULL) {
            dense1->destroy(dense1);
            free(dense1);
        }
        if (relu3 != NULL) {
            relu3->destroy(relu3);
            free(relu3);
        }
        if (dense_residual != NULL) {
            dense_residual->destroy(dense_residual);
            free(dense_residual);
        }
        if (relu_residual != NULL) {
            relu_residual->destroy(relu_residual);
            free(relu_residual);
        }
        if (residual_fc.save != NULL) {
            residual_fc.save->destroy(residual_fc.save);
            free(residual_fc.save);
        }
        if (residual_fc.add != NULL) {
            residual_fc.add->destroy(residual_fc.add);
            free(residual_fc.add);
        }
        if (dense2 != NULL) {
            dense2->destroy(dense2);
            free(dense2);
        }
        if (softmax != NULL) {
            softmax->destroy(softmax);
            free(softmax);
        }
        nn_free(nn);
        return NULL;
    }

    nn_add_layer(nn, conv1);
    nn_add_layer(nn, relu1);
    nn_add_layer(nn, conv2);
    nn_add_layer(nn, relu2);
    nn_add_layer(nn, dropout);
    nn_add_layer(nn, dense1);
    nn_add_layer(nn, relu3);
    if (config->use_residual) {
        nn_add_layer(nn, residual_fc.save);
        nn_add_layer(nn, dense_residual);
        nn_add_layer(nn, relu_residual);
        nn_add_layer(nn, residual_fc.add);
    }
    nn_add_layer(nn, dense2);
    nn_add_layer(nn, softmax);

    return nn;
}

int main(int argc, char **argv)
{
    ExampleConfig config = {
        .train_images = DEFAULT_MNIST_TRAIN_IMAGES,
        .train_labels = DEFAULT_MNIST_TRAIN_LABELS,
        .test_images = DEFAULT_MNIST_TEST_IMAGES,
        .test_labels = DEFAULT_MNIST_TEST_LABELS,
        .limit = 60000,
        .test_limit = -1,
        .epochs = 5,
        .batch_size = 64,
        .val_split = 0.15,
        .learning_rate = 0.001f,
        .weight_decay = 0.0005f,
        .backend = BACKEND_CPU,
        .seed = 42U,
        .log_steps = 100,
        .use_residual = 1,
        .log_memory = 0,
    };

    if (!parse_arguments(argc, argv, &config)) {
        print_usage(argv[0]);
        return 1;
    }

    if (config.log_steps < 0) {
        fprintf(stderr, "log-steps must be non-negative.\n");
        return 1;
    }

#ifdef USE_CUDA
    if (config.backend == BACKEND_GPU) {
        printf("Compiling with CUDA support. Using GPU backend.\n");
    }
#endif

    Dataset *dataset =
        dataset_load_mnist(config.train_images, config.train_labels, config.limit);
    if (dataset == NULL) {
        fprintf(stderr, "Failed to load MNIST training data.\n");
        return 1;
    }

    Dataset *test_dataset = NULL;
    if (config.test_images != NULL && config.test_labels != NULL) {
        test_dataset = dataset_load_mnist(config.test_images, config.test_labels,
                                          config.test_limit);
        if (test_dataset == NULL) {
            fprintf(stderr, "Failed to load MNIST test data.\n");
            dataset_free(dataset);
            return 1;
        }
    }

    Network *nn = build_cnn(dataset, &config);
    if (nn == NULL) {
        dataset_free(dataset);
        if (test_dataset != NULL) {
            dataset_free(test_dataset);
        }
        return 1;
    }

    nn_set_optimizer(nn, OPTIMIZER_ADAMW, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    nn_set_seed(nn, (unsigned long long) config.seed);
    nn_set_log_memory(nn, config.log_memory);

    srand(config.seed);

    int sample_count = dataset->sample_count;
    int val_count = 0;
    if (config.val_split > 0.0 && sample_count > 1) {
        val_count = (int) (config.val_split * sample_count);
        if (val_count < 1) {
            val_count = 1;
        }
        if (val_count >= sample_count) {
            val_count = sample_count - 1;
        }
    }
    int train_count = sample_count - val_count;

    int *indices = (int *) malloc((size_t) sample_count * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "Failed to allocate index buffer.\n");
        nn_free(nn);
        dataset_free(dataset);
        if (test_dataset != NULL) {
            dataset_free(test_dataset);
        }
        return 1;
    }
    for (int i = 0; i < sample_count; i++) {
        indices[i] = i;
    }

    Matrix *batch_inputs = matrix_create(dataset->input_size, config.batch_size);
    Matrix *batch_targets = matrix_create(dataset->output_size, config.batch_size);
    Matrix *batch_predictions = matrix_create(dataset->output_size, config.batch_size);
    Matrix *single_input = matrix_create(dataset->input_size, 1);
    Matrix *single_target = matrix_create(dataset->output_size, 1);

    printf("=== CNN MNIST Example ===\n\n");
    printf("Dataset: mnist (%d samples)\n", sample_count);
    printf("Config:\n");
    printf("  backend:          %s\n", backend_kind_name(config.backend));
    printf("  optimizer:        %s\n", optimizer_kind_name(OPTIMIZER_ADAMW));
    printf("  epochs:           %d\n", config.epochs);
    printf("  batch_size:       %d\n", config.batch_size);
    printf("  learning_rate:    %.6f\n", config.learning_rate);
    printf("  weight_decay:     %.6f\n", config.weight_decay);
    printf("  val_split:        %.4f\n", config.val_split);
    printf("  log_steps:        %d\n", config.log_steps);
    printf("  limit:            %d\n", config.limit);
    printf("  test_limit:       %d\n", config.test_limit);
    printf("  residual:         %s\n", config.use_residual ? "yes" : "no");
    printf("  seed:             %u\n", config.seed);
    printf("  log_memory:       %s\n", config.log_memory ? "yes" : "no");
    printf("  train_images:     %s\n", config.train_images);
    printf("  train_labels:     %s\n", config.train_labels);
    if (config.test_images != NULL && config.test_labels != NULL) {
        printf("  test_images:      %s\n", config.test_images);
        printf("  test_labels:      %s\n", config.test_labels);
    }
    printf("\n");
    printf("Samples: %d (train %d / val %d)\n\n", sample_count, train_count, val_count);
    nn_print_architecture(nn);
    printf("\n");

    if (config.backend == BACKEND_GPU) {
        double flops_per_sample = nn_estimated_flops_per_sample(nn);
        printf("Estimated FLOPs/sample: %.3f GFLOPs\n",
               flops_per_sample / 1e9);
        printf("Estimated FLOPs/step (batch %d): %.3f GFLOPs\n\n", config.batch_size,
               flops_per_sample * (double) config.batch_size / 1e9);
    }

    nn_set_training(nn, 1);

    clock_t train_start_clock = clock();
    long long total_steps = 0;
    double samples_processed = 0.0;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        shuffle_indices(indices, train_count);

        double epoch_loss = 0.0;
        double epoch_grad = 0.0;
        int epoch_correct = 0;
        double epoch_mfu_sum = 0.0;
        size_t last_vram_used_bytes = 0;
        size_t last_vram_total_bytes = 0;

        for (int start = 0; start < train_count; start += config.batch_size) {
            int end = start + config.batch_size;
            if (end > train_count) {
                end = train_count;
            }
            int current_batch = end - start;

            int saved_input_cols = batch_inputs->cols;
            int saved_target_cols = batch_targets->cols;
            int saved_pred_cols = batch_predictions->cols;
            batch_inputs->cols = current_batch;
            batch_targets->cols = current_batch;
            batch_predictions->cols = current_batch;

            for (int col = 0; col < current_batch; col++) {
                int sample_index = indices[start + col];
                dataset_fill_sample_column(dataset, sample_index, batch_inputs,
                                           batch_targets, col);
            }

            nn_zero_gradients(nn);
            TrainStepStats stats = nn_train_batch(nn, batch_inputs, batch_targets,
                                                  current_batch, batch_predictions);
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
            int batch_correct = count_correct_predictions(batch_predictions,
                                                          batch_targets, current_batch);
            epoch_correct += batch_correct;

            total_steps += 1;
            samples_processed += (double) stats.samples;

            if (config.log_steps > 0 &&
                (total_steps == 1 || total_steps % config.log_steps == 0)) {
                clock_t now = clock();
                double elapsed = (double) (now - train_start_clock) / CLOCKS_PER_SEC;
                double samples_per_sec =
                    elapsed > 0.0 ? samples_processed / elapsed : 0.0;
                double batch_accuracy =
                    (double) batch_correct / (double) current_batch * 100.0;
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
                            total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                            (double) stats.mfu * 100.0, vram_used_gb, vram_total_gb, vram_pct,
                            samples_per_sec / 1000.0);
                    } else {
                        printf(
                            "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                            "VRAM: n/a | Samples/s: %.2fK\n",
                            total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                            (double) stats.mfu * 100.0, samples_per_sec / 1000.0);
                    }
                } else {
                    printf(
                        "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | Samples/s: "
                        "%.2fK\n",
                        total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                        samples_per_sec / 1000.0);
                }
            }

            batch_inputs->cols = saved_input_cols;
            batch_targets->cols = saved_target_cols;
            batch_predictions->cols = saved_pred_cols;
        }

        epoch_loss /= (double) train_count;
        double epoch_accuracy = (double) epoch_correct / (double) train_count * 100.0;
        epoch_grad /= (double) train_count;
        double epoch_mfu = train_count > 0 ? epoch_mfu_sum / (double) train_count : 0.0;

        double val_loss = 0.0;
        double val_acc = 0.0;
        if (val_count > 0) {
            nn_set_training(nn, 0);
            evaluate_subset(nn, dataset, &indices[train_count], val_count, single_input,
                            single_target, &val_loss, &val_acc);
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
                printf("Epoch %2d | TrainLoss: %.4f | TrainAcc: %.2f%% | Grad: %.4f | MFU: %.2f%% | "
                       "VRAM: %.2f/%.2f GB (%.1f%%)",
                       epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                       vram_used_gb, vram_total_gb, vram_pct);
            } else {
                printf("Epoch %2d | TrainLoss: %.4f | TrainAcc: %.2f%% | Grad: %.4f | MFU: %.2f%% | "
                       "VRAM: n/a",
                       epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0);
            }
        } else {
            printf("Epoch %2d | TrainLoss: %.4f | TrainAcc: %.2f%% | Grad: %.4f", epoch,
                   epoch_loss, epoch_accuracy, epoch_grad);
        }
        if (val_count > 0) {
            printf(" | ValLoss: %.4f | ValAcc: %.2f%%", val_loss, val_acc);
        }
        printf("\n");
    }

    nn_set_training(nn, 0);

    if (val_count > 0) {
        double val_loss = 0.0;
        double val_acc = 0.0;
        evaluate_subset(nn, dataset, &indices[train_count], val_count, single_input,
                        single_target, &val_loss, &val_acc);
        printf("\nValidation  | Loss: %.4f | Acc: %.2f%%\n", val_loss, val_acc);
    }

    double train_loss = 0.0;
    double train_acc = 0.0;
    evaluate_subset(nn, dataset, indices, train_count, single_input, single_target,
                    &train_loss, &train_acc);
    printf("Train       | Loss: %.4f | Acc: %.2f%%\n", train_loss, train_acc);

    if (test_dataset != NULL) {
        int test_samples = test_dataset->sample_count;
        int *test_indices = (int *) malloc((size_t) test_samples * sizeof(int));
        if (test_indices != NULL) {
            for (int i = 0; i < test_samples; i++) {
                test_indices[i] = i;
            }
            Matrix *test_input = matrix_create(test_dataset->input_size, 1);
            Matrix *test_target = matrix_create(test_dataset->output_size, 1);
            double test_loss = 0.0;
            double test_acc = 0.0;
            evaluate_subset(nn, test_dataset, test_indices, test_samples, test_input,
                            test_target, &test_loss, &test_acc);
            printf("Test        | Loss: %.4f | Acc: %.2f%%\n", test_loss, test_acc);
            matrix_free(test_input);
            matrix_free(test_target);
            free(test_indices);
        }
    }

    matrix_free(batch_inputs);
    matrix_free(batch_targets);
    matrix_free(batch_predictions);
    matrix_free(single_input);
    matrix_free(single_target);
    free(indices);
    nn_free(nn);
    dataset_free(dataset);
    if (test_dataset != NULL) {
        dataset_free(test_dataset);
    }

    return 0;
}
