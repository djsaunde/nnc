#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_CBLAS
void openblas_set_num_threads(int);
#endif

#include "dataset.h"
#include "matrix.h"
#include "net.h"

typedef struct {
    DatasetKind dataset;
    int hidden_size;
    int epochs;
    double learning_rate;
    ActivationKind hidden_activation;
    ActivationKind output_activation;
    double dropout_rate;
    unsigned int seed;
    const char *mnist_images;
    const char *mnist_labels;
    int mnist_limit;
    int batch_size;
    int hidden_specified;
    int output_activation_specified;
    int epochs_specified;
    int learning_rate_specified;
    int batch_specified;
    int log_steps;
    int log_steps_specified;
    double val_split;
    int val_split_specified;
    BackendKind backend;
    int backend_specified;
} TrainingConfig;

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --dataset xor|mnist        Dataset to train on (default: xor)\n");
    printf("  --hidden <int>             Hidden layer size (default: 4 (xor), 128 (mnist))\n");
    printf("  --activation <name>        Hidden activation: sigmoid|relu|tanh (default: sigmoid)\n");
    printf("  --output-activation <name> Output activation: sigmoid|relu|tanh|softmax (default: sigmoid/xor, softmax/mnist)\n");
    printf("  --dropout <float>          Dropout rate in [0,1) for hidden layer (default: 0)\n");
    printf("  --epochs <int>             Training epochs (default: 10000 xor, 5 mnist)\n");
    printf("  --lr <float>               Learning rate (default: 0.5 xor, 0.1 mnist)\n");
    printf("  --seed <int>               RNG seed (default: current time)\n");
    printf("  --batch-size <int>         Minibatch size (default: 1 xor, 32 mnist)\n");
    printf("  --mnist-images <path>      Path to MNIST images file (required for mnist)\n");
    printf("  --mnist-labels <path>      Path to MNIST labels file (required for mnist)\n");
    printf("  --mnist-limit <int>        Optional limit on MNIST samples (default: 1000)\n");
    printf("  --log-steps <int>          Log batch metrics every N steps (default: 1 xor, 100 mnist)\n");
    printf("  --val-split <float>        Validation split ratio (default: 0.15, XOR uses 0)\n");
    printf("  --backend cpu|gpu          Execution backend (default: cpu)\n");
    printf("  --help                     Show this message\n");
}

static int parse_int_arg(const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long result = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || result < INT_MIN || result > INT_MAX) {
        return 0;
    }
    *out = (int) result;
    return 1;
}

static int parse_double_arg(const char *value, double *out)
{
    char *end = NULL;
    errno = 0;
    double result = strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0') {
        return 0;
    }
    *out = result;
    return 1;
}

static int parse_uint_arg(const char *value, unsigned int *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long result = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || result > UINT_MAX) {
        return 0;
    }
    *out = (unsigned int) result;
    return 1;
}

static int parse_activation_arg(const char *value, ActivationKind *out)
{
    if (strcmp(value, "sigmoid") == 0) {
        *out = ACT_SIGMOID;
        return 1;
    }
    if (strcmp(value, "relu") == 0) {
        *out = ACT_RELU;
        return 1;
    }
    if (strcmp(value, "tanh") == 0) {
        *out = ACT_TANH;
        return 1;
    }
    if (strcmp(value, "softmax") == 0) {
        *out = ACT_SOFTMAX;
        return 1;
    }
    return 0;
}

static int parse_dataset_arg(const char *value, DatasetKind *out)
{
    if (strcmp(value, "xor") == 0) {
        *out = DATASET_XOR;
        return 1;
    }
    if (strcmp(value, "mnist") == 0) {
        *out = DATASET_MNIST;
        return 1;
    }
    return 0;
}

static int matrix_argmax(Matrix *m)
{
    double max_value = m->data[0];
    int max_index = 0;
    int total = m->rows * m->cols;
    for (int i = 1; i < total; i++) {
        if (m->data[i] > max_value) {
            max_value = m->data[i];
            max_index = i;
        }
    }
    return max_index;
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

static double compute_loss_from_output(Network *nn, Matrix *output, Matrix *target,
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

static int count_correct_predictions(const Dataset *dataset, Matrix *prediction,
                                     Matrix *target, int batch_size)
{
    int correct = 0;
    if (dataset->kind == DATASET_XOR) {
        for (int col = 0; col < batch_size; col++) {
            int pred_label = prediction->data[col] >= 0.5 ? 1 : 0;
            int true_label = target->data[col] >= 0.5 ? 1 : 0;
            if (pred_label == true_label) {
                correct++;
            }
        }
    } else {
        for (int col = 0; col < batch_size; col++) {
            int pred_label = 0;
            double max_value = prediction->data[col];
            for (int row = 1; row < prediction->rows; row++) {
                double value = prediction->data[row * prediction->cols + col];
                if (value > max_value) {
                    max_value = value;
                    pred_label = row;
                }
            }
            int true_label = 0;
            for (int row = 0; row < target->rows; row++) {
                if (target->data[row * target->cols + col] > 0.5) {
                    true_label = row;
                    break;
                }
            }
            if (pred_label == true_label) {
                correct++;
            }
        }
    }
    return correct;
}

static void evaluate_dataset(Network *nn, const Dataset *dataset)
{
    Matrix *input = matrix_create(dataset->input_size, 1);
    Matrix *target = matrix_create(dataset->output_size, 1);

    if (dataset->kind == DATASET_XOR) {
        printf("=== Testing Trained Network (XOR) ===\n\n");
        for (int i = 0; i < dataset->sample_count; i++) {
            dataset_fill_sample(dataset, i, input, target);
            Matrix *prediction = nn_forward(nn, input);
            printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n", input->data[0],
                   input->data[1], prediction->data[0], target->data[0]);
            matrix_free(prediction);
        }
    } else if (dataset->kind == DATASET_MNIST) {
        printf("=== Evaluating on MNIST (%d samples) ===\n\n", dataset->sample_count);
        int correct = 0;
        for (int i = 0; i < dataset->sample_count; i++) {
            dataset_fill_sample(dataset, i, input, target);
            Matrix *prediction = nn_forward(nn, input);
            int pred = matrix_argmax(prediction);
            int label = dataset_get_label(dataset, i);
            if (pred == label) {
                correct++;
            }
            matrix_free(prediction);
        }
        double accuracy = (double) correct / (double) dataset->sample_count * 100.0;
        printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, dataset->sample_count);
    }

    matrix_free(input);
    matrix_free(target);
}

int main(int argc, char **argv)
{
    TrainingConfig config = {
        .dataset = DATASET_XOR,
        .hidden_size = 4,
        .epochs = 10000,
        .learning_rate = 0.5,
        .hidden_activation = ACT_SIGMOID,
        .output_activation = ACT_SIGMOID,
        .dropout_rate = 0.0,
        .seed = (unsigned int) time(NULL),
        .mnist_images = NULL,
        .mnist_labels = NULL,
        .mnist_limit = 1000,
        .batch_size = 1,
        .hidden_specified = 0,
        .output_activation_specified = 0,
        .epochs_specified = 0,
        .learning_rate_specified = 0,
        .batch_specified = 0,
        .log_steps = 0,
        .log_steps_specified = 0,
        .val_split = 0.15,
#ifdef USE_CUDA
        .val_split_specified = 0,
        .backend = BACKEND_GPU,
#else
        .val_split_specified = 0,
        .backend = BACKEND_CPU,
#endif
        .backend_specified = 0
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--dataset") == 0) {
            if (++i >= argc || !parse_dataset_arg(argv[i], &config.dataset)) {
                fprintf(stderr, "Invalid value for --dataset\n");
                return 1;
            }
        } else if (strcmp(arg, "--hidden") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &config.hidden_size)) {
                fprintf(stderr, "Invalid value for --hidden\n");
                return 1;
            }
            config.hidden_specified = 1;
        } else if (strcmp(arg, "--activation") == 0) {
            if (++i >= argc || !parse_activation_arg(argv[i], &config.hidden_activation)) {
                fprintf(stderr, "Invalid value for --activation\n");
                return 1;
            }
        } else if (strcmp(arg, "--output-activation") == 0) {
            if (++i >= argc || !parse_activation_arg(argv[i], &config.output_activation)) {
                fprintf(stderr, "Invalid value for --output-activation\n");
                return 1;
            }
            config.output_activation_specified = 1;
        } else if (strcmp(arg, "--dropout") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.dropout_rate)) {
                fprintf(stderr, "Invalid value for --dropout\n");
                return 1;
            }
        } else if (strcmp(arg, "--epochs") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &config.epochs)) {
                fprintf(stderr, "Invalid value for --epochs\n");
                return 1;
            }
            config.epochs_specified = 1;
        } else if (strcmp(arg, "--lr") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.learning_rate)) {
                fprintf(stderr, "Invalid value for --lr\n");
                return 1;
            }
            config.learning_rate_specified = 1;
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc || !parse_uint_arg(argv[i], &config.seed)) {
                fprintf(stderr, "Invalid value for --seed\n");
                return 1;
            }
        } else if (strcmp(arg, "--mnist-images") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for --mnist-images\n");
                return 1;
            }
            config.mnist_images = argv[i];
        } else if (strcmp(arg, "--mnist-labels") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for --mnist-labels\n");
                return 1;
            }
            config.mnist_labels = argv[i];
        } else if (strcmp(arg, "--mnist-limit") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &config.mnist_limit)) {
                fprintf(stderr, "Invalid value for --mnist-limit\n");
                return 1;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &config.batch_size)) {
                fprintf(stderr, "Invalid value for --batch-size\n");
                return 1;
            }
            config.batch_specified = 1;
        } else if (strcmp(arg, "--log-steps") == 0) {
            if (++i >= argc || !parse_int_arg(argv[i], &config.log_steps)) {
                fprintf(stderr, "Invalid value for --log-steps\n");
                return 1;
            }
            config.log_steps_specified = 1;
        } else if (strcmp(arg, "--val-split") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.val_split)) {
                fprintf(stderr, "Invalid value for --val-split\n");
                return 1;
            }
            config.val_split_specified = 1;
        } else if (strcmp(arg, "--backend") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for --backend\n");
                return 1;
            }
            if (strcmp(argv[i], "cpu") == 0) {
                config.backend = BACKEND_CPU;
            } else if (strcmp(argv[i], "gpu") == 0) {
                config.backend = BACKEND_GPU;
            } else {
                fprintf(stderr, "Invalid value for --backend\n");
                return 1;
            }
            config.backend_specified = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config.dataset == DATASET_MNIST) {
        if (config.mnist_images == NULL || config.mnist_labels == NULL) {
            fprintf(stderr, "MNIST requires --mnist-images and --mnist-labels.\n");
            return 1;
        }
        if (!config.hidden_specified) {
            config.hidden_size = 128;
        }
        if (!config.epochs_specified) {
            config.epochs = 5;
        }
        if (!config.learning_rate_specified) {
            config.learning_rate = 0.1;
        }
        if (!config.batch_specified) {
            config.batch_size = 32;
        }
        if (!config.output_activation_specified) {
            config.output_activation = ACT_SOFTMAX;
        }
        if (!config.log_steps_specified) {
            config.log_steps = 100;
        }
        if (config.mnist_limit <= 0) {
            config.mnist_limit = 0;
        }
    } else {
        if (!config.batch_specified) {
            config.batch_size = 1;
        }
        if (!config.log_steps_specified) {
            config.log_steps = 1000;
        }
        if (!config.val_split_specified) {
            config.val_split = 0.0;
        }
    }

    if (config.hidden_size <= 0) {
        fprintf(stderr, "Hidden size must be positive.\n");
        return 1;
    }
    if (config.epochs <= 0) {
        fprintf(stderr, "Epoch count must be positive.\n");
        return 1;
    }
    if (config.learning_rate <= 0.0) {
        fprintf(stderr, "Learning rate must be positive.\n");
        return 1;
    }
    if (config.dropout_rate < 0.0 || config.dropout_rate >= 1.0) {
        fprintf(stderr, "Dropout rate must be in [0, 1).\n");
        return 1;
    }
    if (config.batch_size <= 0) {
        fprintf(stderr, "Batch size must be positive.\n");
        return 1;
    }
    if (config.val_split < 0.0 || config.val_split >= 1.0) {
        fprintf(stderr, "Validation split must be in [0, 1).\n");
        return 1;
    }
    if (config.log_steps <= 0) {
        fprintf(stderr, "Log interval must be positive.\n");
        return 1;
    }

    srand(config.seed);

#ifdef USE_CBLAS
    const char *blas_threads_env = getenv("NNC_BLAS_THREADS");
    if (blas_threads_env == NULL) {
        blas_threads_env = getenv("NN_BLAS_THREADS");
    }
    if (blas_threads_env != NULL) {
        int threads = atoi(blas_threads_env);
        if (threads > 0) {
            openblas_set_num_threads(threads);
        }
    } else {
        openblas_set_num_threads(1);
    }
#endif

    Dataset *dataset = NULL;
    if (config.dataset == DATASET_XOR) {
        dataset = dataset_create_xor();
    } else {
        dataset = dataset_load_mnist(config.mnist_images, config.mnist_labels, config.mnist_limit);
    }

    if (dataset == NULL) {
        fprintf(stderr, "Failed to load dataset.\n");
        return 1;
    }

#ifdef USE_CUDA
    char backend_info[256] = "";
    if (config.backend == BACKEND_GPU) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            if (config.backend_specified) {
                fprintf(stderr, "No CUDA-capable device found.\n");
                dataset_free(dataset);
                return 1;
            }
            printf("No CUDA device detected; falling back to CPU backend.\n");
            config.backend = BACKEND_CPU;
        } else {
            int device = 0;
            struct cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            cudaSetDevice(device);
            snprintf(backend_info, sizeof(backend_info), " (device %d: %.200s)", device,
                     prop.name);
        }
    }
#else
    if (config.backend == BACKEND_GPU) {
        fprintf(stderr, "This build was compiled without CUDA support. Rebuild with USE_CUDA=1.\n");
        dataset_free(dataset);
        return 1;
    }
    char backend_info[1] = "";
#endif

    printf("=== Training Neural Network ===\n\n");
    printf("Dataset: %s (%d samples)\n", dataset_kind_name(dataset->kind), dataset->sample_count);
    printf("Config:\n");
    printf("  hidden_size:      %d\n", config.hidden_size);
    printf("  hidden_activation:%s\n", activation_kind_name(config.hidden_activation));
    printf("  output_activation:%s\n", activation_kind_name(config.output_activation));
    printf("  dropout_rate:     %.4f\n", config.dropout_rate);
    printf("  epochs:           %d\n", config.epochs);
    printf("  learning_rate:    %.6f\n", config.learning_rate);
    printf("  batch_size:       %d\n", config.batch_size);
    printf("  log_steps:        %d\n", config.log_steps);
    printf("  val_split:        %.4f\n", config.val_split);
    printf("  backend:          %s%s\n", backend_kind_name(config.backend), backend_info);
    printf("  seed:             %u\n", config.seed);
    if (config.dataset == DATASET_MNIST) {
        printf("  mnist_limit:      %d\n", config.mnist_limit);
        printf("  mnist_images:     %s\n", config.mnist_images);
        printf("  mnist_labels:     %s\n", config.mnist_labels);
    }
    printf("\n");

    Network *nn = nn_create_mlp(dataset->input_size, config.hidden_size, dataset->output_size,
                                config.hidden_activation, config.output_activation,
                                (nn_float) config.dropout_rate, config.backend);
    if (nn == NULL) {
        fprintf(stderr, "Failed to create network.\n");
        dataset_free(dataset);
        return 1;
    }

    int sample_count = dataset->sample_count;
    if (sample_count <= 0) {
        fprintf(stderr, "Dataset is empty.\n");
        nn_free(nn);
        dataset_free(dataset);
        return 1;
    }

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
        return 1;
    }
    for (int i = 0; i < sample_count; i++) {
        indices[i] = i;
    }

    nn_set_training(nn, 1);

    Matrix *batch_inputs = matrix_create(dataset->input_size, config.batch_size);
    Matrix *batch_targets = matrix_create(dataset->output_size, config.batch_size);
    Matrix *batch_prediction = matrix_create(dataset->output_size, config.batch_size);
    Matrix *single_input = matrix_create(dataset->input_size, 1);
    Matrix *single_target = matrix_create(dataset->output_size, 1);

    long long total_steps = 0;
    clock_t epoch_start_clock = clock();

    printf("Training for %d epochs...\n\n", config.epochs);

    int print_interval = (config.dataset == DATASET_XOR) ? 1000 : 1;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        if (config.dataset != DATASET_XOR || val_count > 0) {
            shuffle_indices(indices, sample_count);
        }

        double epoch_loss = 0.0;
        double epoch_grad_sum = 0.0;
        int epoch_correct = 0;

        for (int start = 0; start < train_count; start += config.batch_size) {
            int end = start + config.batch_size;
            if (end > train_count) {
                end = train_count;
            }
            int current_batch = end - start;

            matrix_zero(batch_inputs);
            matrix_zero(batch_targets);

            for (int col = 0; col < current_batch; col++) {
                int sample_index = indices[start + col];
                dataset_fill_sample_column(dataset, sample_index, batch_inputs, batch_targets, col);
            }

            int saved_input_cols = batch_inputs->cols;
            int saved_target_cols = batch_targets->cols;
            int saved_pred_cols = batch_prediction->cols;
            batch_inputs->cols = current_batch;
            batch_targets->cols = current_batch;
            batch_prediction->cols = current_batch;

            nn_zero_gradients(nn);
            TrainStepStats stats =
                nn_train_batch(nn, batch_inputs, batch_targets, current_batch, batch_prediction);
            nn_apply_gradients(nn, (nn_float) config.learning_rate, current_batch);

            batch_inputs->cols = saved_input_cols;
            batch_targets->cols = saved_target_cols;
            batch_prediction->cols = saved_pred_cols;

            epoch_loss += stats.loss * stats.samples;
            epoch_grad_sum += stats.grad_norm * stats.samples;

            int batch_correct =
                count_correct_predictions(dataset, batch_prediction, batch_targets, current_batch);
            epoch_correct += batch_correct;

            total_steps++;
            if (config.log_steps > 0 && (total_steps % config.log_steps == 0 || total_steps == 1)) {
                clock_t now = clock();
                double elapsed = (double) (now - epoch_start_clock) / CLOCKS_PER_SEC;
                double samples_processed = (double) total_steps * current_batch;
                double samples_per_sec = elapsed > 0.0 ? samples_processed / elapsed : 0.0;
                double batch_accuracy = (double) batch_correct / (double) current_batch * 100.0;
                printf("Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | Samples/s: %.2fK\n",
                       total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                       samples_per_sec / 1000.0);
            }
        }

        epoch_loss /= (double) train_count;
        double epoch_accuracy = (double) epoch_correct / (double) train_count * 100.0;
        double epoch_grad = epoch_grad_sum / (double) train_count;

        double val_loss = 0.0;
        double val_accuracy = 0.0;
        if (val_count > 0) {
            nn_set_training(nn, 0);
            int val_correct = 0;
            for (int i = 0; i < val_count; i++) {
                int sample_index = indices[train_count + i];
                dataset_fill_sample(dataset, sample_index, single_input, single_target);
                ForwardCache *val_cache = nn_forward_cached(nn, single_input);
                val_loss += compute_loss_from_output(nn, val_cache->output, single_target, 1);
                if (dataset->kind == DATASET_XOR) {
                    int pred_label = val_cache->output->data[0] >= 0.5 ? 1 : 0;
                    int true_label = single_target->data[0] >= 0.5 ? 1 : 0;
                    if (pred_label == true_label) {
                        val_correct++;
                    }
                } else {
                    int pred_label = matrix_argmax(val_cache->output);
                    int true_label = dataset_get_label(dataset, sample_index);
                    if (pred_label == true_label) {
                        val_correct++;
                    }
                }
                forward_cache_free(val_cache);
            }
            val_loss /= (double) val_count;
            val_accuracy = (double) val_correct / (double) val_count * 100.0;
            nn_set_training(nn, 1);
        }

        if (epoch % print_interval == 0 || epoch == config.epochs - 1) {
            clock_t now = clock();
            double elapsed = (double) (now - epoch_start_clock) / CLOCKS_PER_SEC;
            double samples_processed = (double) total_steps * config.batch_size;
            double samples_per_sec = elapsed > 0.0 ? samples_processed / elapsed : 0.0;
            if (val_count > 0) {
                printf("Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | ValLoss: %.6f | ValAcc: %.2f%% | LR: %.3f | Samples/s: %.2fK\n",
                       epoch, epoch_loss, epoch_accuracy, epoch_grad, val_loss, val_accuracy,
                       config.learning_rate, samples_per_sec / 1000.0);
            } else {
                printf("Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | LR: %.3f | Samples/s: %.2fK\n",
                       epoch, epoch_loss, epoch_accuracy, epoch_grad, config.learning_rate,
                       samples_per_sec / 1000.0);
            }
        }
    }

    printf("\n");

    nn_set_training(nn, 0);
    evaluate_dataset(nn, dataset);

    free(indices);
    matrix_free(batch_inputs);
    matrix_free(batch_targets);
    matrix_free(batch_prediction);
    matrix_free(single_input);
    matrix_free(single_target);
    nn_free(nn);
    dataset_free(dataset);

    return 0;
}
