CC := gcc
CFLAGS := -Wall -Wextra -std=c11 -O3 -march=native -ffast-math -Iinclude -MMD -MP -DUSE_CBLAS

BLAS_LIBS := $(shell pkg-config --libs cblas 2>/dev/null)
BLAS_CFLAGS := $(shell pkg-config --cflags cblas 2>/dev/null)

ifeq ($(BLAS_LIBS),)
BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
endif

ifeq ($(BLAS_LIBS),)
BLAS_LIBS := -lopenblas
BLAS_CFLAGS :=
endif

CFLAGS += $(BLAS_CFLAGS)
LDFLAGS := -lm $(BLAS_LIBS)

SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/nnc

SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

USE_CUDA ?= 0

ifeq ($(USE_CUDA),1)
CFLAGS += -DUSE_CUDA
NVCC := nvcc
NVCCFLAGS := -std=c++14 -O3 -Xcompiler "-Wall -Wextra" -Iinclude -DUSE_CUDA
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS := $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CFLAGS += -I/usr/local/cuda/include
LDFLAGS += -lcudart -lcublas
OBJECTS += $(CUDA_OBJECTS)
else
CUDA_SOURCES :=
CUDA_OBJECTS :=
endif

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(USE_CUDA),1)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

.PHONY: clean run format test

clean:
	rm -f $(OBJECTS) $(TARGET) $(DEPS) $(BUILD_DIR)/nn

run: $(TARGET)
	$(TARGET)

format:
	clang-format -i $(SOURCES) include/*.h

ifeq ($(USE_CUDA),1)
TESTS := $(BUILD_DIR)/compare_backends

$(BUILD_DIR)/compare_backends: tests/compare_backends.c $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))
	$(CC) $(CFLAGS) tests/compare_backends.c $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS)) -o $@ $(LDFLAGS)

test: $(TESTS)
	$(BUILD_DIR)/compare_backends
else
test:
	@echo "USE_CUDA=1 required to run GPU comparison tests"
endif

-include $(DEPS)
