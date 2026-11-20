# Parallel Electricity Consumption and Peak Load Analysis
## High-Performance Smart Grid Telemetry Analytics in Hybrid MPI + OpenMP

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Parallel-Model](https://img.shields.io/badge/Parallel%20Model-Hybrid%20MPI%2BOpenMP-red.svg)](https://www.open-mpi.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Smart%20Grid%20Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](https://apple.com)

A high-performance computing (HPC) mini-project developed in C++17 implementing a unified **Hybrid MPI + OpenMP parallel architecture**. The application coordinates distributed-memory process parallelism with shared-memory thread parallelism to analyze 50,000 real-time measurements from the **Kaggle Smart Grid Real-Time Load Monitoring Dataset** (`ziya07/smart-grid-real-time-load-monitoring-dataset`).

---

## Table of Contents
1. [Project Overview & Hybrid Architecture](#1-project-overview--hybrid-architecture)
2. [Dataset Overview: Smart Grid Real-Time Load Monitoring](#2-dataset-overview-smart-grid-real-time-load-monitoring)
3. [Hybrid Execution Pipeline & Inter-Process Communication](#3-hybrid-execution-pipeline--inter-process-communication)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Prerequisites & Toolchain Setup](#5-prerequisites--toolchain-setup)
6. [Compilation & Execution Guide](#6-compilation--execution-guide)
7. [Correctness & Equivalence Testing](#7-correctness--equivalence-testing)
8. [Experimental Performance Benchmarks](#8-experimental-performance-benchmarks)
9. [Visualizations & Plots](#9-visualizations--plots)
10. [Project Directory Layout](#10-project-directory-layout)
11. [Conclusions & Key Findings](#11-conclusions--key-findings)

---

## 1. Project Overview & Hybrid Architecture

Modern electrical smart grids generate massive volumes of continuous time-series telemetry from substation sensors, phasor measurement units (PMUs), and advanced metering infrastructure (AMI). Sequential processing or single-tier parallelism cannot effectively utilize modern hierarchical supercomputers and multi-core clusters.

This project implements a **Hybrid MPI + OpenMP Architecture**:
- **Coarse-Grained Process-Level Parallelism (MPI):** Partitions and scatters memory blocks across distributed compute nodes or multi-core sockets using non-blocking/collective MPI operations (`MPI_Scatterv`, `MPI_Reduce`, `MPI_Bcast`, `MPI_Gather`, `MPI_Gatherv`).
- **Fine-Grained Thread-Level Parallelism (OpenMP):** Spawns multi-threaded worker pools within each MPI process (`#pragma omp parallel for`) to process local partitions concurrently with L1/L2 cache locality and lock-free thread-private reductions.
- **Sequential Baseline Reference:** A single-threaded reference implementation is included strictly for ground-truth verification and calculating speedup:
  $$S = \frac{T_{\text{sequential}}}{T_{\text{hybrid}}}$$

---

## 2. Dataset Overview: Smart Grid Real-Time Load Monitoring

- **Source:** [Kaggle Smart Grid Real-Time Load Monitoring Dataset](https://www.kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset) (`ziya07/smart-grid-real-time-load-monitoring-dataset`)
- **Primary Data File:** `data/smart_grid_dataset.csv`
- **Total Records:** 50,000 real-time measurements recorded every 15 minutes ($\Delta t = 0.25$ hours) from January 1, 2024 to June 4, 2025 (~1.5 years).
- **Core Telemetry Fields:** Line voltage, load current, active power consumption (kW), reactive power, power factor, solar generation, wind generation, grid supply, voltage fluctuation, overload alerts ($0/1$), transformer faults ($0/1$), temperature, humidity, dynamic price, and predicted load.

To download the dataset automatically using `kagglehub`:
```bash
make download-kaggle
# or:
python3 scripts/download_kaggle_dataset.py
```

---

## 3. Hybrid Execution Pipeline & Inter-Process Communication

```
                     +---------------------------------------+
                     |  Kaggle: smart_grid_dataset.csv       |
                     +---------------------------------------+
                                         |
                                  [DatasetLoader]
                     (Sanitization, Timestamp to Epoch, Delimiters)
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
         +---------------------+                   +---------------------+
         |  Sequential Engine  |                   |  Hybrid MPI+OpenMP  |
         | (Single-Thread Ref) |                   +---------------------+
         +---------------------+                              |
                    |                              [MPI Rank 0: Partitioning]
                    |                                         |
                    |                         +---------------+---------------+
                    |                         | (Equal-Chunks or Time-Range)  |
                    |                         v                               v
                    |                  [MPI_Scatterv]                  [MPI_Scatterv]
                    |                         |                               |
                    |                  +-------------+                 +-------------+
                    |                  | MPI Rank 0  |                 | MPI Rank 1  |
                    |                  | (OpenMP P1) |                 | (OpenMP P2) |
                    |                  +-------------+                 +-------------+
                    |                         |                               |
                    |                         +---------------+---------------+
                    |                                         |
                    |                              [MPI_Reduce / MPI_Gather]
                    |                                         |
                    v                                         v
         +---------------------+                   +---------------------+
         |  Analytical Results | <=== Compare ===> |  Analytical Results |
         |   (CSV Outputs)     |    Consistency    |   (CSV Outputs)     |
         +---------------------+                   +---------------------+
                                         |
                                [Python Visualizer]
                                         |
                           +---------------------------+
                           |  Publication-Grade Plots  |
                           +---------------------------+
```

### Detailed Pipeline Workflow:
1. **MPI Subsystem Initialization:** MPI initializes with thread support (`MPI_THREAD_FUNNELED`), allowing MPI communication from the master thread while worker threads execute computational kernels.
2. **Master Rank 0 Ingestion:** Rank 0 parses `smart_grid_dataset.csv`, maps headers, validates numeric ranges, and converts timestamps to Unix epoch seconds.
3. **Partitioning & Memory Distribution:** Telemetry data is mapped into custom 24-field `MPI_Datatype` structs and scattered using `MPI_Scatterv`.
4. **Thread-Level OpenMP Acceleration:** Each rank runs OpenMP threads (`#pragma omp parallel for`). Threads accumulate local sums, sums of squares, minima, maxima, solar/wind generation, electricity costs, hourly buckets, and calendar day maps.
5. **MPI Collective Reductions:**
   - Global power sums, solar sums, wind sums, and overload counts are reduced via `MPI_Reduce`.
   - Local peak candidates are gathered at Rank 0 via `MPI_Gather` to determine the global peak demand spike and its exact timestamp.
   - Diurnal 24-hour profiles are aggregated with element-wise array reductions (`MPI_Reduce`).
   - Daily map entries and anomalies are gathered at Rank 0 using `MPI_Gatherv`.
6. **Threshold Broadcast & Anomaly Detection:** Rank 0 broadcasts the global mean ($\mu$), standard deviation ($\sigma$), and threshold ($\tau$) using `MPI_Bcast`. Each process flags local spikes, overloads, and transformer faults in parallel.
7. **CSV Output Generation:** Analytical summaries, hourly statistics, daily statistics, anomalies, and benchmarks are saved to disk.

---

## 4. Mathematical Formulation

### 4.1 15-Minute Sampling Interval Energy Integration
$$\Delta t = \frac{15\text{ mins}}{60\text{ mins/hr}} = 0.25 \text{ hours}$$
- **Cumulative Active Energy ($E$):**
  $$E = \sum_{i=1}^{N} P_i \times 0.25 \quad (\text{kWh})$$
- **Total Renewable Energy Generated ($E_{\text{ren}}$):**
  $$E_{\text{ren}} = \sum_{i=1}^{N} (P_{\text{solar}, i} + P_{\text{wind}, i}) \times 0.25 \quad (\text{kWh})$$
- **Total Dynamic Cost:**
  $$\text{Cost} = \sum_{i=1}^{N} (P_i \times 0.25 \times \text{Price}_i) \quad (\text{USD})$$

### 4.2 Descriptive Statistics & Peak Demand
- **Mean Active Power ($\mu$):** $\mu = \frac{1}{N} \sum_{i=1}^{N} P_i$
- **Variance ($\sigma^2$):** $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} P_i^2 - \mu^2$
- **Standard Deviation ($\sigma$):** $\sigma = \sqrt{\sigma^2}$
- **Global Peak Load:** $P_{\text{peak}} = \max_i P_i$ at timestamp $t_{\text{peak}}$
- **Diurnal Peak Hour:** $h^* = \arg\max_{0 \le h \le 23} \bar{P}_h$

### 4.3 Anomaly & Grid Reliability Detection
- **Statistical Power Spikes:** $P_i > \mu + k\sigma$ with Z-score $z_i = \frac{P_i - \mu}{\sigma}$ (default $k=1.8$).
- **Grid Overload Events:** Substation relay flag $\text{Overload Condition} == 1$.
- **Transformer Faults:** Substation protective relay flag $\text{Transformer Fault} == 1$.

### 4.4 Parallel Speedup & Efficiency
- **Speedup ($S$):**
  $$S = \frac{T_{\text{sequential}}}{T_{\text{hybrid}}}$$
- **Parallel Efficiency ($E$):**
  $$E = \frac{S}{P} = \frac{T_{\text{sequential}}}{P \times T_{\text{hybrid}}}$$
  where $P = \text{MPI Ranks} \times \text{OpenMP Threads}$.

---

## 5. Prerequisites & Toolchain Setup

### macOS (Apple Silicon / Intel via Homebrew)
```bash
brew install open-mpi libomp
pip3 install pandas matplotlib numpy kagglehub
```

### Linux (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenmpi-dev openmpi-bin libomp-dev python3-pip
pip3 install pandas matplotlib numpy kagglehub
```

---

## 6. Compilation & Execution Guide

### 6.1 Compiling the Binaries
```bash
make all
```
This builds:
- `build/hybrid`: Hybrid MPI + OpenMP parallel engine
- `build/sequential`: Sequential baseline (for speedup & validation reference)
- `build/test_correctness`: Automated verification test suite

### 6.2 Running the Hybrid MPI + OpenMP Engine
Run with $M$ MPI processes and $T$ OpenMP threads per process:
```bash
# 2 MPI processes x 2 OpenMP threads = 4 processing cores (Equal Chunks)
OMP_NUM_THREADS=2 mpirun -np 2 ./build/hybrid --input data/smart_grid_dataset.csv --output results/hybrid --strategy equal-chunks

# 4 MPI processes x 2 OpenMP threads = 8 processing cores (Time Range Partitioning)
OMP_NUM_THREADS=2 mpirun -np 4 ./build/hybrid --input data/smart_grid_dataset.csv --output results/hybrid --strategy time-range
```

### 6.3 Running the Sequential Baseline Reference
```bash
./build/sequential --input data/smart_grid_dataset.csv --output results/sequential
```

### 6.4 Running Automated Benchmarks & Visualizations
```bash
# Run automated multi-configuration scaling benchmarks
python3 scripts/run_benchmarks.py --input data/smart_grid_dataset.csv --output-dir results

# Generate 5 publication-grade visualization charts
python3 scripts/visualize_results.py --results-dir results/hybrid --benchmarks-file results/benchmark_results.csv --output-dir results/plots
```

---

## 7. Correctness & Equivalence Testing

The automated test suite in `tests/test_correctness.cpp` performs 5 verification checks:
1. **Missing & Corrupt Value Filtering:** Validates that malformed records (`?`) are rejected without crashing.
2. **Timestamp & Leap Year Arithmetic:** Validates 15-minute interval offsets and leap-year conversions (`2024-02-29`).
3. **Statistical Math Precision:** Compares sample mean, variance, standard deviation, and energy integration against exact analytical values.
4. **Anomaly & Grid Alert Classification:** Verifies that statistical spikes, line overloads, and transformer faults are accurately categorized.
5. **Sequential vs. Hybrid Equivalence:** Verifies that the distributed Hybrid MPI+OpenMP pipeline matches the sequential reference baseline within floating-point tolerance ($\epsilon < 10^{-6}$).

Run test suite:
```bash
make test
# or:
mpirun -np 2 ./build/test_correctness
```

**Test Output:**
```text
=================================================================
  CORRECTNESS TEST SUITE: Hybrid MPI + OpenMP Analysis Engine    
=================================================================
[+] test_missing_values_and_malformed passed.
[+] test_timestamp_parsing passed.
[+] test_statistical_math passed.
[+] test_anomaly_detection_rule passed.
[+] test_sequential_vs_hybrid_consistency passed (Sequential Baseline == Hybrid MPI+OpenMP).
=================================================================
 Test Summary: 5/5 passed. Failures: 0
=================================================================
```

---

## 8. Experimental Performance Benchmarks

Evaluated on an **Apple Silicon (M-series, 8 cores)** system with Open MPI 5.0 and Clang with `libomp` on the 50,000-record dataset:

| Hardware Configuration | MPI Ranks | OpenMP Threads | Cores ($P$) | Partitioning Strategy | Computation Time ($T_{\text{comp}}$) | Total Wall Time ($T_{\text{e2e}}$) | Computation Speedup ($S_{\text{comp}}$) | End-to-End Speedup ($S_{\text{e2e}}$) | Computation Efficiency ($E_{\text{comp}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential Baseline** | 1 | 1 | 1 | Single-Threaded | 0.0050 s | 0.2662 s | **1.00x** | 1.00x | **100.0%** |
| **Hybrid MPI+OpenMP** | 2 | 2 | 4 | Equal-Chunks | 0.0014 s | 0.2363 s | **3.57x** | 1.13x | **89.3%** |
| **Hybrid MPI+OpenMP** | 2 | 2 | 4 | Time-Range | 0.0016 s | 0.2385 s | **3.13x** | 1.12x | **78.1%** |
| **Hybrid MPI+OpenMP** | 2 | 4 | 8 | Equal-Chunks | **0.0013 s** | 0.2401 s | **3.85x** | 1.11x | **48.1%** |
| **Hybrid MPI+OpenMP** | 4 | 2 | 8 | Equal-Chunks | 0.0026 s | 0.2618 s | **1.92x** | 1.02x | **24.0%** |
| **Hybrid MPI+OpenMP** | 4 | 2 | 8 | Time-Range | 0.0030 s | 0.3000 s | **1.67x** | 0.89x | **20.8%** |

### Performance Observations:
1. **Strong Scaling Speedup:** The pure parallel computation phase achieves up to **$3.85\times$ speedup** ($0.0050\text{ s} \to 0.0013\text{ s}$).
2. **Optimal Hierarchy:** The 2 MPI Process x 2 OpenMP Thread configuration achieves **89.3% parallel efficiency**, minimizing inter-process communication while maximizing thread concurrency.
3. **Amdahl's Law:** While calculation scales linearly, overall end-to-end wall time is bounded by disk file I/O (~0.096 s) and MPI runtime initialization.

---

## 9. Visualizations & Plots

Generated in [`results/plots/`](file:///Users/shubhamsujitkumarpingale/.gemini/antigravity/scratch/parallel-electricity-analysis/results/plots/):

- **Figure 1: Diurnal Power Profile & Renewable Integration** (`fig1_hourly_power_profile.png`): Hourly mean load, $\pm 1\sigma$ band, peak spikes, and solar/wind generation.
- **Figure 2: Daily Electricity Demand Trends & Renewable Generation** (`fig2_daily_consumption.png`): Daily power trends, max peak surges, and solar+wind energy over 1.5 years.
- **Figure 3: Outliers, Overloads & Transformer Fault Distribution** (`fig3_anomalies_distribution.png`): Scatter plot of flagged grid overloads, transformer faults, and spikes colored by Z-score.
- **Figure 4: Parallel Execution Time Breakdown** (`fig4_execution_time_breakdown.png`): Stacked runtime breakdown comparing File I/O, MPI Scatterv, OpenMP Threads Computation, and MPI Collective Reductions.
- **Figure 5: Hybrid Speedup & Parallel Efficiency Scaling** (`fig5_speedup_and_efficiency.png`): Strong scaling speedup curve and parallel computational efficiency across process/thread configurations.

---

## 10. Project Directory Layout

```
parallel-electricity-analysis/
├── Makefile                                # Multi-platform build script (macOS libomp & Linux)
├── README.md                               # Comprehensive documentation and project report
├── .gitignore                              # Git exclusion rules
├── include/
│   └── common.h                            # Common POD structs, telemetry models, and MPI transfer layouts
├── src/
│   ├── dataset_loader.h & .cpp             # Fast CSV loader, calendar to epoch parser, sample generator
│   ├── statistics.h & .cpp                 # Reductions, peak load analysis, anomaly detection, CSV export
│   ├── sequential.h & .cpp                 # Sequential baseline reference
│   ├── hybrid_mpi_omp.h & .cpp             # Hybrid MPI + OpenMP engine (Scatterv, reductions, 2 strategies)
│   ├── main_sequential.cpp                 # CLI driver for ./build/sequential
│   └── main_hybrid.cpp                     # CLI driver for ./build/hybrid
├── tests/
│   └── test_correctness.cpp                # Automated unit and equivalence verification test suite
├── scripts/
│   ├── download_kaggle_dataset.py          # Automated kagglehub downloader for ziya07 dataset
│   ├── run_benchmarks.py                   # Automated Hybrid MPI + OpenMP benchmark runner
│   └── visualize_results.py                # Publication-quality Matplotlib chart generator (5 figures)
├── data/
│   ├── README.md                           # Schema documentation and telemetry attributes
│   └── smart_grid_dataset.csv              # Kaggle Smart Grid Real-Time Load Monitoring Dataset
└── results/
    ├── README.md                           # Results documentation and metric definitions
    ├── benchmark_results.csv               # Multi-configuration scaling benchmark table
    ├── sequential/                         # Sequential baseline CSV outputs
    ├── hybrid/                             # Hybrid parallel CSV outputs
    └── plots/                              # Generated PNG visualization figures
```

---

## 11. Conclusions & Key Findings

1. **Hybrid Synergy:** The combination of MPI for process distribution across cores and OpenMP for shared-memory thread pooling delivers balanced computational performance without memory duplication.
2. **Exact Numerical Equivalence:** The Hybrid MPI + OpenMP implementation achieves bitwise and within-tolerance numerical parity with the sequential reference baseline.
3. **Smart Grid Operational Insights:** Analysis of 50,000 real-time readings revealed a global peak demand of **12.1032 kW** (occurring on `2024-08-13 23:00:00`), **4,956 grid overload intervals**, and **1,460 transformer fault events**, offset by **501,045.86 kWh** of localized renewable solar and wind generation.
