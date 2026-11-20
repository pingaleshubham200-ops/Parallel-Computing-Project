# Makefile for Parallel Electricity Consumption and Peak Load Analysis
# Hybrid MPI + OpenMP Parallel Architecture
# Dataset: Kaggle Smart Grid Real-Time Load Monitoring Dataset
# Platform: macOS (Apple Silicon & Intel) and Linux

CXX ?= clang++
MPICXX ?= mpicxx
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Iinclude -Isrc

# Detect Operating System and Configure OpenMP Flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS Configuration
    ifeq ($(shell test -d /opt/homebrew/opt/libomp && echo yes),yes)
        # Apple Silicon Homebrew
        OMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
    else ifeq ($(shell test -d /usr/local/opt/libomp && echo yes),yes)
        # Intel Mac Homebrew
        OMP_FLAGS = -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp
    else
        OMP_FLAGS = -Xpreprocessor -fopenmp -lomp
    endif
else
    # Linux Configuration (GCC / Clang)
    OMP_FLAGS = -fopenmp
endif

BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests

# Common object files
COMMON_SRCS = $(SRC_DIR)/dataset_loader.cpp $(SRC_DIR)/statistics.cpp $(SRC_DIR)/sequential.cpp
COMMON_OBJS = $(BUILD_DIR)/dataset_loader.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/sequential.o

# Hybrid MPI + OpenMP object
HYBRID_OBJS = $(BUILD_DIR)/hybrid_mpi_omp.o

# Executables
TARGET_SEQ    = $(BUILD_DIR)/sequential
TARGET_HYBRID = $(BUILD_DIR)/hybrid
TARGET_TEST   = $(BUILD_DIR)/test_correctness

DATASET_FILE = data/smart_grid_dataset.csv

.PHONY: all clean test sample-data visualize benchmark download-kaggle help

all: directories $(TARGET_SEQ) $(TARGET_HYBRID) $(TARGET_TEST)

directories:
	@mkdir -p $(BUILD_DIR) data results/sequential results/hybrid results/plots

# Object compilation
$(BUILD_DIR)/dataset_loader.o: $(SRC_DIR)/dataset_loader.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/statistics.o: $(SRC_DIR)/statistics.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/sequential.o: $(SRC_DIR)/sequential.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/hybrid_mpi_omp.o: $(SRC_DIR)/hybrid_mpi_omp.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@

# 1. Sequential Baseline Executable (for validation & reference speedup)
$(TARGET_SEQ): directories $(COMMON_OBJS) $(BUILD_DIR)/main_sequential.o
	$(CXX) $(CXXFLAGS) $(COMMON_OBJS) $(BUILD_DIR)/main_sequential.o -o $@
	@echo "[+] Built Sequential Reference: $@"

$(BUILD_DIR)/main_sequential.o: $(SRC_DIR)/main_sequential.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 2. Hybrid MPI + OpenMP Executable
$(TARGET_HYBRID): directories $(COMMON_OBJS) $(HYBRID_OBJS) $(BUILD_DIR)/main_hybrid.o
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) $(COMMON_OBJS) $(HYBRID_OBJS) $(BUILD_DIR)/main_hybrid.o -o $@
	@cp -f $@ $(BUILD_DIR)/hybrid_parallel 2>/dev/null || true
	@echo "[+] Built Hybrid MPI+OpenMP Parallel Engine: $@"

$(BUILD_DIR)/main_hybrid.o: $(SRC_DIR)/main_hybrid.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@

# 3. Correctness Test Suite (Validates Sequential vs Hybrid MPI+OpenMP)
$(TARGET_TEST): directories $(COMMON_OBJS) $(HYBRID_OBJS) $(BUILD_DIR)/test_correctness.o
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) $(COMMON_OBJS) $(HYBRID_OBJS) $(BUILD_DIR)/test_correctness.o -o $@
	@echo "[+] Built Hybrid Correctness Test Suite: $@"

$(BUILD_DIR)/test_correctness.o: $(TEST_DIR)/test_correctness.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@

download-kaggle:
	@python3 scripts/download_kaggle_dataset.py

sample-data: $(TARGET_SEQ)
	@echo "[*] Generating synthetic sample dataset for verification..."
	@./$(TARGET_SEQ) --generate-sample --input data/sample_smart_grid_dataset.csv

test: $(TARGET_TEST)
	@echo "[*] Executing Hybrid MPI + OpenMP correctness tests with 2 MPI processes..."
	@mpirun -np 2 ./$(TARGET_TEST)

run-seq: $(TARGET_SEQ)
	@./$(TARGET_SEQ) --input $(DATASET_FILE) --output results/sequential

run-hybrid: $(TARGET_HYBRID)
	@OMP_NUM_THREADS=2 mpirun -np 2 ./$(TARGET_HYBRID) --input $(DATASET_FILE) --output results/hybrid --strategy equal-chunks

visualize:
	@python3 scripts/visualize_results.py --results-dir results/hybrid --benchmarks-file results/benchmark_results.csv --output-dir results/plots

benchmark: $(TARGET_SEQ) $(TARGET_HYBRID)
	@python3 scripts/run_benchmarks.py --input $(DATASET_FILE) --output-dir results

clean:
	rm -rf $(BUILD_DIR)/*
	@echo "[+] Cleaned build directory."

help:
	@echo "Available Targets:"
	@echo "  all              - Build Sequential Reference, Hybrid MPI+OpenMP Engine, and Test suite"
	@echo "  download-kaggle  - Download Kaggle dataset using kagglehub"
	@echo "  sample-data      - Generate synthetic test dataset"
	@echo "  test             - Run test suite comparing Sequential vs Hybrid MPI+OpenMP"
	@echo "  run-seq          - Run sequential baseline analysis"
	@echo "  run-hybrid       - Run Hybrid MPI + OpenMP parallel analysis"
	@echo "  visualize        - Generate publication-grade matplotlib charts"
	@echo "  benchmark        - Run multi-core Hybrid MPI + OpenMP scaling benchmarks"
	@echo "  clean            - Remove all build artifacts"
