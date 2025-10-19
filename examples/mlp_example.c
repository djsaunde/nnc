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
    int use_residual;
    unsigned int seed;
    const char *mnist_images;
    const char *mnist_labels;
    const char *mnist_test_images;
    const char *mnist_test_labels;
    int mnist_limit;
    int batch_size;
    int hidden_specified;
    int hidden_activation_specified;
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
    OptimizerKind optimizer;
    int optimizer_specified;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int beta1_specified;
    int beta2_specified;
    int epsilon_specified;
    int weight_decay_specified;
    int log_memory;
} TrainingConfig;

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --dataset xor|mnist        Dataset to train on (default: mnist)\n");
    printf(
        "  --hidden <int>             Hidden layer size (default: 128 mnist, 4 xor)\n");
    printf(
        "  --activation <name>        Hidden activation: sigmoid|relu|tanh (default: "
        "relu mnist, sigmoid xor)\n");
    printf(
        "  --output-activation <name> Output activation: sigmoid|relu|tanh|softmax "
        "(default: softmax mnist, sigmoid xor)\n");
    printf(
        "  --dropout <float>          Dropout rate in [0,1) for hidden layer (default: "
        "0)\n");
    printf(
        "  --epochs <int>             Training epochs (default: 5 mnist, 10000 xor)\n");
    printf(
        "  --lr <float>               Learning rate (default: 0.1 mnist, 0.5 xor)\n");
    printf("  --seed <int>               RNG seed (default: current time)\n");
    printf("  --batch-size <int>         Minibatch size (default: 32 mnist, 1 xor)\n");
    printf(
        "  --mnist-images <path>      Path to MNIST images file (default: "
        "datasets/mnist/train-images.idx3-ubyte)\n");
    printf(
        "  --mnist-labels <path>      Path to MNIST labels file (default: "
        "datasets/mnist/train-labels.idx1-ubyte)\n");
    printf(
        "  --mnist-test-images <path> Path to MNIST test images file (default: "
        "datasets/mnist/t10k-images.idx3-ubyte)\n");
    printf(
        "  --mnist-test-labels <path> Path to MNIST test labels file (default: "
        "datasets/mnist/t10k-labels.idx1-ubyte)\n");
    printf(
        "  --mnist-limit <int>        Optional limit on MNIST samples (default: 60000, "
        "0 means all)\n");
    printf(
        "  --log-steps <int>          Log batch metrics every N steps (0 = epoch only; "
        "default: 100 mnist, 1000 xor)\n");
    printf(
        "  --val-split <float>        Validation split ratio (default: 0.15 mnist, 0 "
        "xor)\n");
    printf("  --log-memory              Log GPU memory usage each step (default: off)\n");
    printf(
        "  --optimizer <name>         Optimizer: sgd|adamw (default: adamw mnist, sgd "
        "xor)\n");
    printf("  --beta1 <float>            AdamW beta1 (default: 0.9)\n");
    printf("  --beta2 <float>            AdamW beta2 (default: 0.999)\n");
    printf("  --epsilon <float>          AdamW epsilon (default: 1e-8)\n");
    printf("  --weight-decay <float>     AdamW weight decay (default: 0.01 mnist)\n");
    printf("  --backend cpu|gpu          Execution backend (default: cpu)\n");
    printf("  --help                     Show this message\n");
}

static int parse_int_arg(const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long result = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || result < INT_MIN ||
        result > INT_MAX) {
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

static int parse_optimizer_arg(const char *value, OptimizerKind *out)
{
    if (strcmp(value, "sgd") == 0) {
        *out = OPTIMIZER_SGD;
        return 1;
    }
    if (strcmp(value, "adamw") == 0) {
        *out = OPTIMIZER_ADAMW;
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
            printf("Input: [%.1f, %.1f] -> Output: %.4f (Expected: %.1f)\n",
                   input->data[0], input->data[1], prediction->data[0],
                   target->data[0]);
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

typedef struct {
    double loss;
    double accuracy;
} SubsetMetrics;

static Network *build_mlp_network(const Dataset *dataset, const TrainingConfig *config)
{
    if (!config->use_residual) {
        return nn_create_mlp(dataset->input_size, config->hidden_size,
                             dataset->output_size, config->hidden_activation,
                             config->output_activation, (nn_float) config->dropout_rate,
                             config->backend);
    }

    Network *nn =
        nn_create_empty(dataset->input_size, dataset->output_size, config->backend);
    if (nn == NULL) {
        return NULL;
    }

    Layer *dense_ih = layer_dense_create_backend(config->backend, dataset->input_size,
                                                 config->hidden_size);
    Layer *act_hidden =
        layer_activation_from_kind_backend(config->backend, config->hidden_activation);
    Layer *dropout = NULL;
    if (config->dropout_rate > 0.0) {
        dropout = layer_dropout_create((nn_float) config->dropout_rate);
    }
    SkipConnection residual = skip_connection_create();
    Layer *dense_residual = NULL;
    Layer *act_residual = NULL;
    if (residual.save == NULL || residual.add == NULL) {
        if (residual.save != NULL) {
            residual.save->destroy(residual.save);
            free(residual.save);
        }
        if (residual.add != NULL) {
            residual.add->destroy(residual.add);
            free(residual.add);
        }
        nn_free(nn);
        return NULL;
    }
    dense_residual = layer_dense_create_backend(config->backend, config->hidden_size,
                                                config->hidden_size);
    act_residual =
        layer_activation_from_kind_backend(config->backend, config->hidden_activation);

    Layer *dense_ho = layer_dense_create_backend(config->backend, config->hidden_size,
                                                 dataset->output_size);
    Layer *act_output =
        layer_activation_from_kind_backend(config->backend, config->output_activation);

    if (dense_ih == NULL || act_hidden == NULL ||
        (config->dropout_rate > 0.0 && dropout == NULL) || dense_residual == NULL ||
        act_residual == NULL || dense_ho == NULL || act_output == NULL) {
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
        if (dense_residual != NULL) {
            dense_residual->destroy(dense_residual);
            free(dense_residual);
        }
        if (act_residual != NULL) {
            act_residual->destroy(act_residual);
            free(act_residual);
        }
        if (dense_ho != NULL) {
            dense_ho->destroy(dense_ho);
            free(dense_ho);
        }
        if (act_output != NULL) {
            act_output->destroy(act_output);
            free(act_output);
        }
        residual.save->destroy(residual.save);
        free(residual.save);
        residual.add->destroy(residual.add);
        free(residual.add);
        nn_free(nn);
        return NULL;
    }

    nn_add_layer(nn, dense_ih);
    nn_add_layer(nn, act_hidden);
    if (dropout != NULL) {
        nn_add_layer(nn, dropout);
    }
    nn_add_layer(nn, residual.save);
    nn_add_layer(nn, dense_residual);
    nn_add_layer(nn, act_residual);
    nn_add_layer(nn, residual.add);
    nn_add_layer(nn, dense_ho);
    nn_add_layer(nn, act_output);

    return nn;
}

static SubsetMetrics compute_subset_metrics(Network *nn, const Dataset *dataset,
                                            const int *indices, int count,
                                            Matrix *input, Matrix *target)
{
    SubsetMetrics metrics = {.loss = 0.0, .accuracy = 0.0};
    if (count <= 0) {
        return metrics;
    }

    int correct = 0;
    for (int i = 0; i < count; i++) {
        int sample_index = indices[i];
        dataset_fill_sample(dataset, sample_index, input, target);
        ForwardCache *cache = nn_forward_cached(nn, input);
        metrics.loss += compute_loss_from_output(nn, cache->output, target, 1);

        if (dataset->kind == DATASET_XOR) {
            int pred_label = cache->output->data[0] >= 0.5 ? 1 : 0;
            int true_label = target->data[0] >= 0.5 ? 1 : 0;
            if (pred_label == true_label) {
                correct++;
            }
        } else {
            int pred_label = matrix_argmax(cache->output);
            int true_label = dataset_get_label(dataset, sample_index);
            if (pred_label == true_label) {
                correct++;
            }
        }

        forward_cache_free(cache);
    }

    metrics.loss /= (double) count;
    metrics.accuracy = (double) correct / (double) count * 100.0;
    return metrics;
}

int main(int argc, char **argv)
{
    TrainingConfig config = {
        .dataset = DATASET_MNIST,
        .hidden_size = 128,
        .epochs = 5,
        .learning_rate = 0.001,
        .hidden_activation = ACT_RELU,
        .output_activation = ACT_SOFTMAX,
        .dropout_rate = 0.0,
        .use_residual = 0,
        .seed = (unsigned int) time(NULL),
        .mnist_images = "datasets/mnist/train-images.idx3-ubyte",
        .mnist_labels = "datasets/mnist/train-labels.idx1-ubyte",
        .mnist_test_images = "datasets/mnist/t10k-images.idx3-ubyte",
        .mnist_test_labels = "datasets/mnist/t10k-labels.idx1-ubyte",
        .mnist_limit = 60000,
        .batch_size = 32,
        .hidden_specified = 0,
        .hidden_activation_specified = 0,
        .output_activation_specified = 0,
        .epochs_specified = 0,
        .learning_rate_specified = 0,
        .batch_specified = 0,
        .log_steps = 100,
        .log_steps_specified = 0,
        .val_split = 0.15,
        .val_split_specified = 0,
#ifdef USE_CUDA
        .backend = BACKEND_GPU,
#else
        .backend = BACKEND_CPU,
#endif
        .backend_specified = 0,
        .optimizer = OPTIMIZER_ADAMW,
        .optimizer_specified = 0,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .weight_decay = 0.01,
        .beta1_specified = 0,
        .beta2_specified = 0,
        .epsilon_specified = 0,
        .weight_decay_specified = 0,
        .log_memory = 0};

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
            if (++i >= argc ||
                !parse_activation_arg(argv[i], &config.hidden_activation)) {
                fprintf(stderr, "Invalid value for --activation\n");
                return 1;
            }
            config.hidden_activation_specified = 1;
        } else if (strcmp(arg, "--output-activation") == 0) {
            if (++i >= argc ||
                !parse_activation_arg(argv[i], &config.output_activation)) {
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
        } else if (strcmp(arg, "--mnist-test-images") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for --mnist-test-images\n");
                return 1;
            }
            config.mnist_test_images = argv[i];
        } else if (strcmp(arg, "--mnist-test-labels") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for --mnist-test-labels\n");
                return 1;
            }
            config.mnist_test_labels = argv[i];
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
        } else if (strcmp(arg, "--residual") == 0) {
            config.use_residual = 1;
        } else if (strcmp(arg, "--no-residual") == 0) {
            config.use_residual = 0;
        } else if (strcmp(arg, "--val-split") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.val_split)) {
                fprintf(stderr, "Invalid value for --val-split\n");
                return 1;
            }
            config.val_split_specified = 1;
        } else if (strcmp(arg, "--optimizer") == 0) {
            if (++i >= argc || !parse_optimizer_arg(argv[i], &config.optimizer)) {
                fprintf(stderr, "Invalid value for --optimizer\n");
                return 1;
            }
            config.optimizer_specified = 1;
        } else if (strcmp(arg, "--beta1") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.beta1)) {
                fprintf(stderr, "Invalid value for --beta1\n");
                return 1;
            }
            config.beta1_specified = 1;
        } else if (strcmp(arg, "--beta2") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.beta2)) {
                fprintf(stderr, "Invalid value for --beta2\n");
                return 1;
            }
            config.beta2_specified = 1;
        } else if (strcmp(arg, "--epsilon") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.epsilon)) {
                fprintf(stderr, "Invalid value for --epsilon\n");
                return 1;
            }
            config.epsilon_specified = 1;
        } else if (strcmp(arg, "--weight-decay") == 0) {
            if (++i >= argc || !parse_double_arg(argv[i], &config.weight_decay)) {
                fprintf(stderr, "Invalid value for --weight-decay\n");
                return 1;
            }
            config.weight_decay_specified = 1;
        } else if (strcmp(arg, "--log-memory") == 0) {
            config.log_memory = 1;
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
        if (!config.hidden_activation_specified) {
            config.hidden_activation = ACT_RELU;
        }
        if (!config.epochs_specified) {
            config.epochs = 5;
        }
        if (!config.learning_rate_specified) {
            config.learning_rate = 0.001;
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
        if (!config.val_split_specified) {
            config.val_split = 0.15;
        }
        if (!config.optimizer_specified) {
            config.optimizer = OPTIMIZER_ADAMW;
        }
        if (!config.beta1_specified) {
            config.beta1 = 0.9;
        }
        if (!config.beta2_specified) {
            config.beta2 = 0.999;
        }
        if (!config.epsilon_specified) {
            config.epsilon = 1e-8;
        }
        if (!config.weight_decay_specified) {
            config.weight_decay = 0.01;
        }
        if (config.optimizer == OPTIMIZER_SGD && !config.learning_rate_specified) {
            config.learning_rate = 0.1;
        }
    } else {
        if (!config.hidden_specified) {
            config.hidden_size = 4;
        }
        if (!config.hidden_activation_specified) {
            config.hidden_activation = ACT_SIGMOID;
        }
        if (!config.output_activation_specified) {
            config.output_activation = ACT_SIGMOID;
        }
        if (!config.epochs_specified) {
            config.epochs = 10000;
        }
        if (!config.learning_rate_specified) {
            config.learning_rate = 0.5;
        }
        if (!config.batch_specified) {
            config.batch_size = 1;
        }
        if (!config.log_steps_specified) {
            config.log_steps = 1000;
        }
        if (!config.val_split_specified) {
            config.val_split = 0.0;
        }
        if (!config.optimizer_specified) {
            config.optimizer = OPTIMIZER_SGD;
        }
        if (!config.weight_decay_specified) {
            config.weight_decay = 0.0;
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
    if (config.log_steps < 0) {
        fprintf(stderr, "Log interval must be non-negative.\n");
        return 1;
    }
    if (config.weight_decay < 0.0) {
        fprintf(stderr, "Weight decay must be non-negative.\n");
        return 1;
    }
    if (config.optimizer == OPTIMIZER_ADAMW) {
        if (config.beta1 <= 0.0 || config.beta1 >= 1.0) {
            fprintf(stderr, "beta1 must be in (0, 1).\n");
            return 1;
        }
        if (config.beta2 <= 0.0 || config.beta2 >= 1.0) {
            fprintf(stderr, "beta2 must be in (0, 1).\n");
            return 1;
        }
        if (config.epsilon <= 0.0) {
            fprintf(stderr, "epsilon must be positive.\n");
            return 1;
        }
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
        dataset = dataset_load_mnist(config.mnist_images, config.mnist_labels,
                                     config.mnist_limit);
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
        fprintf(
            stderr,
            "This build was compiled without CUDA support. Rebuild with USE_CUDA=1.\n");
        dataset_free(dataset);
        return 1;
    }
    char backend_info[1] = "";
#endif

    printf("=== Training Neural Network ===\n\n");
    printf("Dataset: %s (%d samples)\n", dataset_kind_name(dataset->kind),
           dataset->sample_count);
    printf("Config:\n");
    printf("  hidden_size:      %d\n", config.hidden_size);
    printf("  hidden_activation:%s\n", activation_kind_name(config.hidden_activation));
    printf("  output_activation:%s\n", activation_kind_name(config.output_activation));
    printf("  dropout_rate:     %.4f\n", config.dropout_rate);
    printf("  epochs:           %d\n", config.epochs);
    printf("  learning_rate:    %.6f\n", config.learning_rate);
    printf("  optimizer:        %s\n", optimizer_kind_name(config.optimizer));
    printf("  beta1:            %.6f\n", config.beta1);
    printf("  beta2:            %.6f\n", config.beta2);
    printf("  epsilon:          %.6e\n", config.epsilon);
    printf("  weight_decay:     %.6f\n", config.weight_decay);
    printf("  batch_size:       %d\n", config.batch_size);
    printf("  log_steps:        %d\n", config.log_steps);
    printf("  residual:         %s\n", config.use_residual ? "yes" : "no");
    printf("  val_split:        %.4f\n", config.val_split);
    printf("  backend:          %s%s\n", backend_kind_name(config.backend),
           backend_info);
    printf("  seed:             %u\n", config.seed);
    printf("  log_memory:       %s\n", config.log_memory ? "yes" : "no");
    if (config.dataset == DATASET_MNIST) {
        printf("  mnist_limit:      %d\n", config.mnist_limit);
        printf("  mnist_images:     %s\n", config.mnist_images);
        printf("  mnist_labels:     %s\n", config.mnist_labels);
        printf("  mnist_test_images:%s\n", config.mnist_test_images);
        printf("  mnist_test_labels:%s\n", config.mnist_test_labels);
    }
    printf("\n");

    Network *nn = build_mlp_network(dataset, &config);
    if (nn == NULL) {
        fprintf(stderr, "Failed to create network.\n");
        dataset_free(dataset);
        return 1;
    }

    nn_set_optimizer(nn, config.optimizer, (nn_float) config.beta1,
                     (nn_float) config.beta2, (nn_float) config.epsilon,
                     (nn_float) config.weight_decay);
    nn_set_seed(nn, config.seed);
    nn_set_log_memory(nn, config.log_memory);

    nn_print_architecture(nn);
    printf("\n");

    if (config.backend == BACKEND_GPU) {
        double flops_per_sample = nn_estimated_flops_per_sample(nn);
        printf("Estimated FLOPs/sample: %.3f GFLOPs\n",
               flops_per_sample / 1e9);
        printf("Estimated FLOPs/step (batch %d): %.3f GFLOPs\n\n", config.batch_size,
               flops_per_sample * (double) config.batch_size / 1e9);
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

    if (sample_count > 1) {
        shuffle_indices(indices, sample_count);
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
        shuffle_indices(indices, train_count);

        double epoch_loss = 0.0;
        double epoch_grad_sum = 0.0;
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

            matrix_zero(batch_inputs);
            matrix_zero(batch_targets);

            for (int col = 0; col < current_batch; col++) {
                int sample_index = indices[start + col];
                dataset_fill_sample_column(dataset, sample_index, batch_inputs,
                                           batch_targets, col);
            }

            int saved_input_cols = batch_inputs->cols;
            int saved_target_cols = batch_targets->cols;
            int saved_pred_cols = batch_prediction->cols;
            batch_inputs->cols = current_batch;
            batch_targets->cols = current_batch;
            batch_prediction->cols = current_batch;

            nn_zero_gradients(nn);
            TrainStepStats stats = nn_train_batch(nn, batch_inputs, batch_targets,
                                                  current_batch, batch_prediction);
            nn_apply_gradients(nn, (nn_float) config.learning_rate, current_batch);

            batch_inputs->cols = saved_input_cols;
            batch_targets->cols = saved_target_cols;
            batch_prediction->cols = saved_pred_cols;

            epoch_loss += stats.loss * stats.samples;
            epoch_grad_sum += stats.grad_norm * stats.samples;
            epoch_mfu_sum += (double) stats.mfu * stats.samples;
            if (config.backend == BACKEND_GPU) {
                last_vram_used_bytes = stats.vram_used_bytes;
                if (stats.vram_total_bytes > 0) {
                    last_vram_total_bytes = stats.vram_total_bytes;
                }
            }

            int batch_correct = count_correct_predictions(dataset, batch_prediction,
                                                          batch_targets, current_batch);
            epoch_correct += batch_correct;

            total_steps++;
            if (config.log_steps > 0 &&
                (total_steps % config.log_steps == 0 || total_steps == 1)) {
                clock_t now = clock();
                double elapsed = (double) (now - epoch_start_clock) / CLOCKS_PER_SEC;
                double samples_processed = (double) total_steps * current_batch;
                double samples_per_sec =
                    elapsed > 0.0 ? samples_processed / elapsed : 0.0;
                double batch_accuracy =
                    (double) batch_correct / (double) current_batch * 100.0;
                if (config.backend == BACKEND_GPU) {
                    if (config.log_memory) {
                        if (stats.vram_total_bytes > 0) {
                            double vram_total_gb =
                                (double) stats.vram_total_bytes /
                                (1024.0 * 1024.0 * 1024.0);
                            double vram_used_gb =
                                (double) stats.vram_used_bytes /
                                (1024.0 * 1024.0 * 1024.0);
                            double vram_pct = (double) stats.vram_used_bytes /
                                              (double) stats.vram_total_bytes * 100.0;
                            printf(
                                "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | "
                                "MFU: %.2f%% | VRAM: %.2f/%.2f GB (%.1f%%) | Samples/s: %.2fK\n",
                                total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                                (double) stats.mfu * 100.0, vram_used_gb, vram_total_gb,
                                vram_pct, samples_per_sec / 1000.0);
                        } else {
                            printf(
                                "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | "
                                "MFU: %.2f%% | VRAM: n/a | Samples/s: %.2fK\n",
                                total_steps, stats.loss, batch_accuracy, stats.grad_norm,
                                (double) stats.mfu * 100.0, samples_per_sec / 1000.0);
                        }
                    } else {
                        printf(
                            "Step %lld | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                            "Samples/s: %.2fK\n",
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
        }

        epoch_loss /= (double) train_count;
        double epoch_accuracy = (double) epoch_correct / (double) train_count * 100.0;
        double epoch_grad = epoch_grad_sum / (double) train_count;
        double epoch_mfu = train_count > 0 ? epoch_mfu_sum / (double) train_count : 0.0;

        double val_loss = 0.0;
        double val_accuracy = 0.0;
        if (val_count > 0) {
            nn_set_training(nn, 0);
            int val_correct = 0;
            for (int i = 0; i < val_count; i++) {
                int sample_index = indices[train_count + i];
                dataset_fill_sample(dataset, sample_index, single_input, single_target);
                ForwardCache *val_cache = nn_forward_cached(nn, single_input);
                val_loss +=
                    compute_loss_from_output(nn, val_cache->output, single_target, 1);
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
            if (config.backend == BACKEND_GPU) {
                if (!config.log_memory) {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "ValLoss: %.6f | ValAcc: %.2f%% | LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        val_loss, val_accuracy, config.learning_rate,
                        samples_per_sec / 1000.0);
                } else if (last_vram_total_bytes > 0) {
                    double vram_total_gb =
                        (double) last_vram_total_bytes / (1024.0 * 1024.0 * 1024.0);
                    double vram_used_gb =
                        (double) last_vram_used_bytes / (1024.0 * 1024.0 * 1024.0);
                    double vram_pct = (double) last_vram_used_bytes /
                                      (double) last_vram_total_bytes * 100.0;
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "VRAM: %.2f/%.2f GB (%.1f%%) | ValLoss: %.6f | ValAcc: %.2f%% | "
                        "LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        vram_used_gb, vram_total_gb, vram_pct, val_loss, val_accuracy,
                        config.learning_rate, samples_per_sec / 1000.0);
                } else {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "VRAM: n/a | ValLoss: %.6f | ValAcc: %.2f%% | LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        val_loss, val_accuracy, config.learning_rate,
                        samples_per_sec / 1000.0);
                }
            } else {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | ValLoss: %.6f "
                        "| ValAcc: %.2f%% | LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, val_loss,
                        val_accuracy, config.learning_rate, samples_per_sec / 1000.0);
                }
            } else {
            if (config.backend == BACKEND_GPU) {
                if (!config.log_memory) {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        config.learning_rate, samples_per_sec / 1000.0);
                } else if (last_vram_total_bytes > 0) {
                    double vram_total_gb =
                        (double) last_vram_total_bytes / (1024.0 * 1024.0 * 1024.0);
                    double vram_used_gb =
                        (double) last_vram_used_bytes / (1024.0 * 1024.0 * 1024.0);
                    double vram_pct = (double) last_vram_used_bytes /
                                      (double) last_vram_total_bytes * 100.0;
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "VRAM: %.2f/%.2f GB (%.1f%%) | LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        vram_used_gb, vram_total_gb, vram_pct, config.learning_rate,
                        samples_per_sec / 1000.0);
                } else {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | MFU: %.2f%% | "
                        "VRAM: n/a | LR: %.3f | Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad, epoch_mfu * 100.0,
                        config.learning_rate, samples_per_sec / 1000.0);
                }
            } else {
                    printf(
                        "Epoch %5d | Loss: %.6f | Acc: %.2f%% | Grad: %.6f | LR: %.3f | "
                        "Samples/s: %.2fK\n",
                        epoch, epoch_loss, epoch_accuracy, epoch_grad,
                        config.learning_rate, samples_per_sec / 1000.0);
                }
            }
        }
    }

    printf("\n");

    nn_set_training(nn, 0);

    if (train_count > 0) {
        SubsetMetrics train_metrics = compute_subset_metrics(
            nn, dataset, indices, train_count, single_input, single_target);
        printf("Final Train | Loss: %.6f | Acc: %.2f%%\n", train_metrics.loss,
               train_metrics.accuracy);
    }
    if (val_count > 0) {
        SubsetMetrics val_metrics = compute_subset_metrics(
            nn, dataset, indices + train_count, val_count, single_input, single_target);
        printf("Final Val   | Loss: %.6f | Acc: %.2f%%\n", val_metrics.loss,
               val_metrics.accuracy);
    }

    if (dataset->kind == DATASET_MNIST && config.mnist_test_images != NULL &&
        config.mnist_test_labels != NULL) {
        Dataset *test_dataset =
            dataset_load_mnist(config.mnist_test_images, config.mnist_test_labels, 0);
        if (test_dataset != NULL) {
            int test_count = test_dataset->sample_count;
            if (test_count > 0) {
                int *test_indices = (int *) malloc((size_t) test_count * sizeof(int));
                if (test_indices != NULL) {
                    for (int i = 0; i < test_count; i++) {
                        test_indices[i] = i;
                    }
                    Matrix *test_input = matrix_create(test_dataset->input_size, 1);
                    Matrix *test_target = matrix_create(test_dataset->output_size, 1);
                    SubsetMetrics test_metrics =
                        compute_subset_metrics(nn, test_dataset, test_indices,
                                               test_count, test_input, test_target);
                    printf("Final Test  | Loss: %.6f | Acc: %.2f%%\n",
                           test_metrics.loss, test_metrics.accuracy);
                    matrix_free(test_input);
                    matrix_free(test_target);
                    free(test_indices);
                } else {
                    fprintf(stderr, "Failed to allocate test indices.\n");
                }
            }
            dataset_free(test_dataset);
        } else {
            fprintf(stderr,
                    "Failed to load MNIST test dataset (images: %s, labels: %s).\n",
                    config.mnist_test_images, config.mnist_test_labels);
        }
    }

    if (dataset->kind == DATASET_XOR && train_count > 0) {
        evaluate_dataset(nn, dataset);
    }

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
